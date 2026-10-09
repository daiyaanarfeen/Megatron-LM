// Diagnostic GPU-only activity collection. No API callbacks or kernel replay.
#include <cupti.h>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if CUPTI_API_VERSION >= 130400
using KernelRecord = CUpti_ActivityKernel13;
using MemcpyRecord = CUpti_ActivityMemcpy7;
using MemsetRecord = CUpti_ActivityMemset5;
#else
using KernelRecord = CUpti_ActivityKernel11;
using MemcpyRecord = CUpti_ActivityMemcpy6;
using MemsetRecord = CUpti_ActivityMemset4;
#endif

struct Event {
    uint64_t start, end;
    uint32_t stream, device, correlation;
    std::string name, category;
    uint64_t bytes = 0;
};
static std::vector<Event> events;
static std::mutex event_mutex;
static size_t dropped = 0;
static uint64_t origin = 0;
static bool whole_graph = false;
static bool memory_activity = true;
static bool eager_only = false;
static bool minimal_full = false;
#if CUPTI_API_VERSION >= 130400
static CUpti_SubscriberHandle subscriber = nullptr;
static std::map<uint32_t, std::map<int, CUpti_ActivityFieldLayoutEntry>> record_layouts;
#endif
static std::map<uint32_t, size_t> record_kinds;
struct CompletedBuffer {
    CUcontext context;
    uint32_t stream;
    uint8_t* data;
    size_t valid;
};
static constexpr size_t buffer_bytes = 8 * 1024 * 1024;
static std::array<uint8_t*, 32> buffer_pool;
static std::atomic<size_t> next_buffer{0};
static std::vector<CompletedBuffer> completed_buffers;

static void checked(CUptiResult status, int line) {
    if (status != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(status, &message);
        fprintf(stderr, "GPU_TRACE_CUPTI_ERROR line=%d %s\n", line, message);
        std::abort();
    }
}
#define check(status) checked(status, __LINE__)

static void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* records) {
    *size = buffer_bytes;
    *records = 0;
    auto index = next_buffer.fetch_add(1);
    if (index >= buffer_pool.size()) std::abort();
    *buffer = buffer_pool[index];
}

static void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t* buffer,
                             size_t size, size_t valid) {
    std::lock_guard<std::mutex> lock(event_mutex);
    completed_buffers.push_back({context, stream, buffer, valid});
}

#if CUPTI_API_VERSION >= 130400
static void CUPTIAPI request_v2(uint8_t** buffer, size_t* size, size_t* records,
                               CUpti_BufferCallbackRequestInfo*) {
    request(buffer, size, records);
}
static void CUPTIAPI complete_v2(uint8_t* buffer, size_t size, size_t valid,
                                CUpti_BufferCallbackCompleteInfo* info) {
    std::lock_guard<std::mutex> lock(event_mutex);
    for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
                      CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_GRAPH_TRACE}) {
        if (!record_layouts[kind].empty() || !info->ppRecordLayouts || !info->ppRecordLayouts[kind]) continue;
        auto* layout = info->ppRecordLayouts[kind];
        for (size_t i = 0; i < layout->numFields; ++i) {
            record_layouts[kind].emplace(layout->pEntries[i].fieldId, layout->pEntries[i]);
        }
    }
    completed_buffers.push_back({nullptr, 0, buffer, valid});
}
static void CUPTIAPI callback(void*, CUpti_CallbackDomain, CUpti_CallbackId, const void*) {}
template<typename T> static T field(CUpti_Activity* record, int id) {
    const auto& layout = record_layouts.at(record->kind).at(id);
    if (layout.size != sizeof(T)) std::abort();
    T value;
    memcpy(&value, reinterpret_cast<uint8_t*>(record) + layout.offset, sizeof(T));
    return value;
}
#endif

static CUptiResult next_record(uint8_t* buffer, size_t valid, CUpti_Activity** record) {
#if CUPTI_API_VERSION >= 130400
    if (eager_only) return cuptiActivityGetNextRecord_v2(subscriber, buffer, valid, record);
#endif
    return cuptiActivityGetNextRecord(buffer, valid, record);
}

static void activity(CUpti_ActivityKind kind, bool enable) {
#if CUPTI_API_VERSION >= 130400
    if (eager_only) {
        int graph_fields[] = {GRAPH_TRACE_FIELD_KIND, GRAPH_TRACE_FIELD_START, GRAPH_TRACE_FIELD_END,
                             GRAPH_TRACE_FIELD_STREAM_ID, GRAPH_TRACE_FIELD_DEVICE_ID};
        int kernel_fields[] = {KERNEL_FIELD_KIND, KERNEL_FIELD_START, KERNEL_FIELD_END,
                               KERNEL_FIELD_STREAM_ID, KERNEL_FIELD_DEVICE_ID, KERNEL_FIELD_NAME};
        int copy_fields[] = {MEMCPY_FIELD_KIND, MEMCPY_FIELD_START, MEMCPY_FIELD_END,
                            MEMCPY_FIELD_STREAM_ID, MEMCPY_FIELD_DEVICE_ID, MEMCPY_FIELD_BYTES};
        int set_fields[] = {MEMSET_FIELD_KIND, MEMSET_FIELD_START, MEMSET_FIELD_END,
                           MEMSET_FIELD_STREAM_ID, MEMSET_FIELD_DEVICE_ID, MEMSET_FIELD_BYTES};
        CUpti_ActivityConfig config = {sizeof(CUpti_ActivityConfig)};
        config.fieldSelection = {sizeof(CUpti_ActivityFieldSelection),
                                 kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE ? (minimal_full ? 5UL : 1UL) : 6UL,
                                 kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE ? graph_fields :
                                 kind == CUPTI_ACTIVITY_KIND_MEMCPY ? copy_fields :
                                 kind == CUPTI_ACTIVITY_KIND_MEMSET ? set_fields : kernel_fields};
        check(enable ? cuptiActivityEnable_v2(subscriber, kind, &config)
                     : cuptiActivityDisable_v2(subscriber, kind, nullptr));
        return;
    }
#endif
    check(enable ? cuptiActivityEnable(kind) : cuptiActivityDisable(kind));
}

static void parse_buffer(CUcontext context, uint32_t stream, uint8_t* buffer, size_t valid) {
    std::vector<Event> batch;
    CUpti_Activity* record = nullptr;
    CUptiResult status;
    while ((status = next_record(buffer, valid, &record)) == CUPTI_SUCCESS) {
        ++record_kinds[record->kind];
#if CUPTI_API_VERSION >= 130400
        if (eager_only) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                batch.push_back({field<uint64_t>(record, KERNEL_FIELD_START), field<uint64_t>(record, KERNEL_FIELD_END),
                                 field<uint32_t>(record, KERNEL_FIELD_STREAM_ID), field<uint32_t>(record, KERNEL_FIELD_DEVICE_ID),
                                 0, field<const char*>(record, KERNEL_FIELD_NAME), "kernel"});
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                batch.push_back({field<uint64_t>(record, MEMCPY_FIELD_START), field<uint64_t>(record, MEMCPY_FIELD_END),
                                 field<uint32_t>(record, MEMCPY_FIELD_STREAM_ID), field<uint32_t>(record, MEMCPY_FIELD_DEVICE_ID),
                                 0, "Memcpy", "gpu_memcpy", field<uint64_t>(record, MEMCPY_FIELD_BYTES)});
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
                batch.push_back({field<uint64_t>(record, MEMSET_FIELD_START), field<uint64_t>(record, MEMSET_FIELD_END),
                                 field<uint32_t>(record, MEMSET_FIELD_STREAM_ID), field<uint32_t>(record, MEMSET_FIELD_DEVICE_ID),
                                 0, "Memset", "gpu_memset", field<uint64_t>(record, MEMSET_FIELD_BYTES)});
            } else if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE && minimal_full) {
                batch.push_back({field<uint64_t>(record, GRAPH_TRACE_FIELD_START), field<uint64_t>(record, GRAPH_TRACE_FIELD_END),
                                 field<uint32_t>(record, GRAPH_TRACE_FIELD_STREAM_ID), field<uint32_t>(record, GRAPH_TRACE_FIELD_DEVICE_ID),
                                 0, "CUDA graph", "graph_span"});
            }
            continue;
        }
#endif
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto* r = reinterpret_cast<KernelRecord*>(record);
            batch.push_back({r->start, r->end, r->streamId, r->deviceId,
                             r->correlationId, r->name, "kernel"});
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* r = reinterpret_cast<MemcpyRecord*>(record);
            batch.push_back({r->start, r->end, r->streamId, r->deviceId,
                             r->correlationId, "Memcpy", "gpu_memcpy", r->bytes});
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto* r = reinterpret_cast<MemsetRecord*>(record);
            batch.push_back({r->start, r->end, r->streamId, r->deviceId,
                             r->correlationId, "Memset", "gpu_memset", r->bytes});
        } else if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
            auto* r = reinterpret_cast<CUpti_ActivityGraphTrace2*>(record);
            batch.push_back({r->start, r->end, r->streamId, r->deviceId,
                             r->correlationId, "CUDA graph " + std::to_string(r->graphId), "graph_span"});
        }
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) check(status);
    size_t lost = 0;
#if CUPTI_API_VERSION >= 130400
    if (eager_only) check(cuptiActivityGetNumDroppedRecords_v2(subscriber, context, stream, &lost));
    else
#endif
    check(cuptiActivityGetNumDroppedRecords(context, stream, &lost));
    {
        std::lock_guard<std::mutex> lock(event_mutex);
        dropped += lost;
        events.insert(events.end(), std::make_move_iterator(batch.begin()),
                      std::make_move_iterator(batch.end()));
    }
}

extern "C" void gpu_trace_init() {
    uint32_t version = 0;
    check(cuptiGetVersion(&version));
    fprintf(stderr, "GPU_TRACE_CUPTI_VERSION %u headers=%u\n", version, CUPTI_API_VERSION);
    uint8_t zeroed = 1;
    size_t zeroed_size = sizeof(zeroed);
    const char* eager = std::getenv("CUPTI_EAGER_ONLY");
    eager_only = eager && std::string(eager) == "1";
    const char* full = std::getenv("CUPTI_MINIMAL_FULL");
    minimal_full = full && std::string(full) == "1";
#if CUPTI_API_VERSION >= 130400
    if (eager_only) {
        CUpti_SubscriberParams params = {sizeof(CUpti_SubscriberParams), "NEP GPU trace", nullptr, 0, 1};
        check(cuptiSubscribe_v2(&subscriber, callback, nullptr, &params));
        check(cuptiActivitySetAttribute_v2(subscriber, CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS,
                                         &zeroed_size, &zeroed));
        check(cuptiActivitySetAttribute_v2(subscriber, CUPTI_ACTIVITY_ATTR_ENABLE_MULTI_SUBSCRIBER_GRAPH_TRACE,
                                         &zeroed_size, &zeroed));
        check(cuptiActivitySetAttribute_v2(subscriber, CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
                                         &zeroed_size, &zeroed));
    } else
#else
    if (eager_only) std::abort();
#endif
    check(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
                                   &zeroed_size, &zeroed));
    const char* preallocate = std::getenv("CUPTI_PREALLOCATE_DEVICE_POOL");
    if (preallocate && std::string(preallocate) == "1") {
        for (auto setting : {
                 std::make_pair(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, size_t(16 * 1024 * 1024)),
                 std::make_pair(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, size_t(64)),
                 std::make_pair(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, size_t(16))}) {
            size_t bytes = sizeof(setting.second);
#if CUPTI_API_VERSION >= 130400
            if (eager_only)
                check(cuptiActivitySetAttribute_v2(subscriber, setting.first, &bytes, &setting.second));
            else
#endif
                check(cuptiActivitySetAttribute(setting.first, &bytes, &setting.second));
        }
        fprintf(stderr, "GPU_TRACE_DEVICE_POOL preallocate=16 buffer_bytes=16777216 limit=64\n");
    }
    for (auto& buffer : buffer_pool) {
        if (posix_memalign(reinterpret_cast<void**>(&buffer), 64, buffer_bytes)) std::abort();
        memset(buffer, 0, buffer_bytes);
    }
    completed_buffers.reserve(buffer_pool.size());
#if CUPTI_API_VERSION >= 130400
    if (eager_only) check(cuptiActivityRegisterCallbacks_v2(subscriber, request_v2, complete_v2));
    else
#endif
    check(cuptiActivityRegisterCallbacks(request, complete));
    events.reserve(250000);
    const char* graph = std::getenv("CUPTI_WHOLE_GRAPH");
    whole_graph = !eager_only && graph && std::string(graph) == "1";
    const char* memory = std::getenv("CUPTI_MEMORY_ACTIVITY");
    memory_activity = !memory || std::string(memory) != "0";
    if (eager_only && memory_activity && !minimal_full) std::abort();
    fprintf(stderr, "GPU_TRACE_MEMORY_ACTIVITY %d\n", memory_activity);
    fprintf(stderr, "GPU_TRACE_EAGER_ONLY %d\n", eager_only);
    fprintf(stderr, "GPU_TRACE_MINIMAL_FULL %d\n", minimal_full);
}

extern "C" void gpu_trace_prepare() {
    const char* late = std::getenv("CUPTI_LATE_GRAPH_ACTIVITY");
    if (late && std::string(late) == "1") return;
    // Enable after selecting HES, but before application graph instantiation.
    if (whole_graph || eager_only) activity(CUPTI_ACTIVITY_KIND_GRAPH_TRACE, true);
}

extern "C" uint64_t gpu_trace_timestamp() {
    uint64_t timestamp;
#if CUPTI_API_VERSION >= 130400
    if (eager_only) check(cuptiGetTimestamp_v2(subscriber, &timestamp));
    else
#endif
    check(cuptiGetTimestamp(&timestamp));
    return timestamp;
}

extern "C" void gpu_trace_enable_hes() {
#if CUPTI_API_VERSION >= 130400
    if (eager_only) {
        uint8_t enabled = 1;
        size_t size = sizeof(enabled);
        check(cuptiActivitySetAttribute_v2(subscriber, CUPTI_ACTIVITY_ATTR_ENABLE_HES,
                                         &size, &enabled));
    } else
#endif
    check(cuptiActivityEnableHWTrace(1));
    fprintf(stderr, "GPU_TRACE_HES enabled\n");
}

extern "C" void gpu_trace_start() {
    origin = gpu_trace_timestamp();
    const char* late = std::getenv("CUPTI_LATE_GRAPH_ACTIVITY");
    if (late && std::string(late) == "1" && (whole_graph || eager_only))
        activity(CUPTI_ACTIVITY_KIND_GRAPH_TRACE, true);
    for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                      CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET}) {
        if (!memory_activity && kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
        activity(kind, true);
    }
}

extern "C" void gpu_trace_range(const char* name, uint64_t start, uint64_t end) {
    std::lock_guard<std::mutex> lock(event_mutex);
    events.push_back({start, end, 0, 0, 0, name, "capture_range"});
}

extern "C" void gpu_trace_graph_span(uint64_t start, uint64_t end) {
    std::lock_guard<std::mutex> lock(event_mutex);
    events.push_back({start, end, 0, 0, 0, "CUDA graph (event envelope)", "graph_span"});
}

extern "C" void gpu_trace_stop(const char* path) {
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                      CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET}) {
        if (!memory_activity && kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
        activity(kind, false);
    }
    if (whole_graph || eager_only) activity(CUPTI_ACTIVITY_KIND_GRAPH_TRACE, false);
    check(cuptiActivityFlushAll(0));
    for (const auto& buffer : completed_buffers) {
        parse_buffer(buffer.context, buffer.stream, buffer.data, buffer.valid);
    }
    std::lock_guard<std::mutex> lock(event_mutex);
    std::ofstream out(path);
    if (!out) std::abort();
    out << "{\"origin_ns\":" << origin << ",\"dropped_records\":" << dropped
        << ",\"memory_activity\":" << (memory_activity ? "true" : "false")
        << ",\"eager_only\":" << (eager_only ? "true" : "false")
        << ",\"traceEvents\":[";
    bool first = true;
    size_t invalid = 0;
    size_t prewindow = 0;
    size_t zero_graphs = 0;
    for (const auto& e : events) {
        if (e.end <= origin && e.start > 0 && e.end > e.start) { ++prewindow; continue; }
        if (e.start < origin || e.end < e.start || (e.end == e.start && e.category != "graph_span")) {
            if (++invalid <= 4) fprintf(stderr, "GPU_TRACE_INVALID category=%s start=%llu end=%llu origin=%llu\n",
                                       e.category.c_str(), (unsigned long long)e.start,
                                       (unsigned long long)e.end, (unsigned long long)origin);
            continue;
        }
        if (e.end == e.start) ++zero_graphs;
        if (!first) out << ',';
        first = false;
        out << "{\"ph\":\"X\",\"cat\":" << std::quoted(e.category)
            << ",\"name\":" << std::quoted(e.name)
            << ",\"ts\":" << std::fixed << std::setprecision(3) << (e.start-origin)/1000.0
            << ",\"dur\":" << (e.end-e.start)/1000.0
            << ",\"pid\":" << (e.category == "capture_range" ? 1 : 0)
            << ",\"tid\":" << e.stream
            << ",\"args\":{\"stream\":" << e.stream << ",\"device\":" << e.device
            << ",\"correlation\":" << e.correlation << ",\"bytes\":" << e.bytes << "}}";
    }
    out << "],\"invalid_records\":" << invalid << ",\"prewindow_records\":" << prewindow
        << ",\"zero_duration_graphs\":" << zero_graphs << "}\n";
    fprintf(stderr, "GPU_TRACE_RESULT events=%zu dropped=%zu invalid=%zu\n", events.size(), dropped, invalid);
    for (const auto& kind : record_kinds) {
        fprintf(stderr, "GPU_TRACE_KIND %u count=%zu\n", kind.first, kind.second);
    }
}
