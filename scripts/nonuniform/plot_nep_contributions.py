"""Render architecture diagrams; no benchmark data or runtime modifications."""

import ast
import math
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


OUT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_contributions_20260911')
SOURCE = Path('/home/darfeen/Megatron-LM-nep-main-20260817/megatron/core/distributed/nonuniform_common.py')
INK = '#18252d'
MUTED = '#53636c'
BLUE = '#166da3'
GREEN = '#267b56'


def font(size, bold=False):
    suffix = '-Bold' if bold else ''
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf', size)


def label(draw, xy, text, size=28, color=INK, bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


def box(draw, rect, title, lines=(), native=False, size=27):
    x, y, right, bottom = rect
    color = GREEN if native else BLUE
    draw.rounded_rectangle(rect, radius=8, fill='#edf6f0' if native else '#edf5fb', outline=color, width=3)
    rows = [(title, True)] + [(line, False) for line in lines]
    for j, (line, bold) in enumerate(rows):
        f = font(size, bold)
        assert draw.textlength(line, font=f) <= right - x - 32, line
        assert y + 16 + j * (size + 12) + size + 8 < bottom, line
        label(draw, (x + 16, y + 16 + j * (size + 12)), line, size, color if bold else INK, bold)


def arrow(draw, start, end, color=BLUE, both=False):
    draw.line([start, end], fill=color, width=5)
    for tip, tail in [(end, start)] + ([(start, end)] if both else []):
        angle = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
        points = [tip] + [(tip[0] - 17 * math.cos(angle + a), tip[1] - 17 * math.sin(angle + a)) for a in [-0.5, 0.5]]
        draw.polygon(points, fill=color)


def save(im, name):
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f'{name}.png')
    im.save(OUT / f'{name}.pdf', resolution=160)


def verify_layout():
    names = {'build_expert_to_ep_rank_map', 'compute_nonuniform_ep_owner_expert_slots',
             'compute_nonuniform_ep_dispatch_slots', 'compute_nonuniform_ep_expert_placement'}
    tree = ast.parse(SOURCE.read_text())
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = dict(math=math, List=List, Optional=Optional, Sequence=Sequence)
    exec(compile(selected, str(SOURCE), 'exec'), namespace)
    place = namespace['compute_nonuniform_ep_expert_placement']
    assert place(10, 4, 3) == [[0, 1, 2], [4, 5, 6], [7, 8], [3, 9]]
    assert place(10, 3, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for e, n, k in [(128, 16, 12), (128, 32, 28), (512, 16, 12), (128, 8, 4)]:
        p = place(e, n, k)
        assert sorted(sum(p, [])) == list(range(e))
        assert max(map(len, p)) - min(map(len, p)) <= 1


def systems():
    im = Image.new('RGB', (2400, 1430), 'white')
    d = ImageDraw.Draw(im)
    label(d, (70, 40), 'NEP systems: one logical model, different physical layouts', 46, bold=True)
    label(d, (70, 108), 'Illustration: 10 real experts | EP4 / EP3 | expert TP1 | 3 matching logical owners', 30, MUTED)
    for x, right, title in [(70, 1030, 'Full replica: NVLink domain A'), (1370, 2330, 'Reduced replica: NVLink domain B')]:
        d.rounded_rectangle((x, 200, right, 980), radius=8, outline='#a7b3ba', width=3)
        label(d, (x + 30, 225), title, 32, bold=True)
        label(d, (x + 30, 540), 'Canonical owner communication layouts', 29, bold=True)
    for i, experts in enumerate(['0, 1, 2', '4, 5, 6', '7, 8', '3, 9']):
        box(d, (95 + 230*i, 300, 310 + 230*i, 435), f'GPU {i}', [experts, 'follower' if i == 3 else 'owner'], size=24)
    for i, experts in enumerate(['0, 1, 2, 3', '4, 5, 6', '7, 8, 9']):
        box(d, (1400 + 300*i, 300, 1680 + 300*i, 435), f'GPU {i}: owner', [experts], size=25)
    arrow(d, (350, 442), (350, 524))
    label(d, (105, 465), 'Gather', 25, BLUE)
    arrow(d, (760, 524), (760, 442))
    label(d, (785, 450), 'Parameter', 22, BLUE)
    label(d, (785, 480), 'redistribution', 22, BLUE)
    label(d, (1430, 465), 'Same placement as owners: local staging only', 27, MUTED)
    for i, experts in enumerate(['0  1  2  3', '4  5  6  EMPTY', '7  8  9  EMPTY']):
        y = 605 + i*115
        for x in [120, 1450]:
            box(d, (x, y, x+830, y+90), f'Owner {i}:  experts {experts}', size=28)
        arrow(d, (960, y+48), (1440, y+48), GREEN, both=True)
        label(d, (1060, y+4), f'EDP group {i}', 24, GREEN, True)
        label(d, (1085, y+65), 'RDMA', 22, GREEN)
    label(d, (100, 935), 'EMPTY: layout padding only; no dummy expert parameters', 25, MUTED)
    label(d, (70, 1015), 'Distributed optimizer lifecycle: dependencies, not an overlap timeline', 31, bold=True)
    entries = [('NEP Gather', ['Physical grads', 'to owner layout'], False),
               ('Reduce-scatter', ['Native EDP', 'gradient sync'], True),
               ('Optimizer step', ['Native sharded', 'owner state'], True),
               ('Parameter AG', ['Native EDP', 'all-gather'], True),
               ('Redistribution', ['Owner parameters', 'to physical holders'], False)]
    for i, (title, lines, native) in enumerate(entries):
        x = 70 + i*475
        box(d, (x, 1080, x+410, 1240), title, lines, native, 27)
        if i < 4:
            arrow(d, (x+415, 1160), (x+465, 1160), MUTED)
    label(d, (70, 1280), 'Inherited from NTP: common-owner synchronization. NEP: expert-ID layouts + virtual slots + native optimizer bridge.', 27)
    label(d, (70, 1330), 'Optimizer state is sharded across each EDP group. Blue = NEP-specific adapter; green = native Megatron operation.', 27, MUTED)
    save(im, 'nep_systems')


def implementation():
    im = Image.new('RGB', (2400, 1500), 'white')
    d = ImageDraw.Draw(im)
    label(d, (70, 40), 'NEP implementation: expert adapters around native Megatron', 44, bold=True)
    label(d, (70, 108), 'NonuniformEPDistributedDataParallel extends native DistributedDataParallel', 30, MUTED)
    label(d, (70, 158), 'Blue: NEP extension    Green: existing Megatron machinery    Arrows: call/dependency order', 27, MUTED)
    rows = [
        ('1. Construction and expert integration', 260, [
            ('NEP topology', ['Assign real experts;', 'build virtual slots', 'before model construction.'], False),
            ('Native DDP.__init__', ['Physical parameter buffers;', 'dense/expert groups;', 'autograd + forward hooks.'], True),
            ('NEP expert adapter', ['Replace expert group views;', 'update param-to-group map;', 'allocate owner buffers.'], False),
            ('Native optimizer factory', ['Use owner proxy parameters', 'and native buffers for', 'expert optimizer state.'], True)]),
        ('2. Backward: existing hook, expert-specific synchronization', 560, [
            ('Native AccumulateGrad', ['Accumulate parameter grad;', 'final GA marks group ready', 'through existing lookup.'], True),
            ('NEP register_grad_ready', ['Track canonical readiness;', 'preserve bucket and', 'participant ordering.'], False),
            ('NEP Gather', ['Physical -> owner layout;', 'stage native EDP buffer;', 'enforce Gather dependency.'], False),
            ('Native start_grad_sync', ['Owner EDP group performs', 'reduce-scatter for DistOpt;', 'all-reduce otherwise.'], True)]),
        ('3. Optimizer and next forward: native parameter-sync hook', 900, [
            ('Native DistOpt.step', ['Update sharded owner state;', 'followers hold no expert', 'optimizer state.'], True),
            ('Native parameter sync', ['All-gather owner parameters;', 'existing prefetch and', 'completion machinery.'], True),
            ('NEP finish_param_sync', ['Redistribute updated params', 'to physical expert holders', 'before layer consumption.'], False),
            ('Native MoE forward', ['Flex / HybridEP dispatch;', 'real local expert GEMMs;', 'dense path unchanged.'], True)])]
    for heading, y, cells in rows:
        label(d, (70, y-55), heading, 31, bold=True)
        for i, (title, lines, native) in enumerate(cells):
            x = 70+i*590
            box(d, (x, y, x+500, y+210), title, lines, native, 25)
            if i < 3:
                arrow(d, (x+510, y+105), (x+580, y+105), MUTED)
    d.rectangle((70, 802, 2330, 842), fill='#edf6f0')
    label(d, (90, 805), 'Dense parameters retain native buffers, bucket groups, gradient synchronization, and optimization.', 26, GREEN)
    box(d, (70, 1190, 1165, 1380), 'Native machinery retained', ['No second per-parameter autograd hook.', 'Native collectives and optimizer algorithms are reused.', 'Owner buffers participate in native zero/scale/finalize lifecycle.'], True, 27)
    box(d, (1210, 1190, 2330, 1380), 'NEP responsibilities', ['Canonical expert keys and real/virtual slot mappings.', 'Physical <-> owner transfers and buffer-lifetime safety.', 'Ordering and completion before optimizer/layer consumption.'], False, 27)
    label(d, (70, 1420), 'Forward nesting: native pre-hook -> NEP finish_param_sync -> native AG completion -> NEP redistribution.', 27, MUTED)
    save(im, 'nep_implementation')


def expert_contributions():
    im = Image.new('RGB', (2400, 1640), 'white')
    d = ImageDraw.Draw(im)
    label(d, (70, 35), 'NEP: expert identity survives changes in replica width', 46, bold=True)
    label(d, (70, 105), 'Illustrative 10-expert model, EP4 / EP3. Same real experts and router; different physical layouts.', 29, MUTED)
    label(d, (70, 180), '1. Expert-aware ownership: synchronize by expert ID, not local parameter offset', 34, bold=True)
    box(d, (70, 255, 730, 470), 'Full replica: EP4 physical holders', ['GPU0: E0 E1 E2    GPU1: E4 E5 E6', 'GPU2: E7 E8         GPU3: E3 E9', 'E3 lives on follower GPU3.'], size=27)
    box(d, (1670, 255, 2330, 470), 'Reduced replica: EP3 physical holders', ['GPU0: E0 E1 E2 E3', 'GPU1: E4 E5 E6    GPU2: E7 E8 E9', 'E3 lives on owner GPU0.'], size=27)
    box(d, (835, 270, 1565, 455), 'Why native local-offset pairing fails', ['EP4 GPU0 has 3 experts; EP3 GPU0 has 4.', 'Matching local buckets does not identify', 'the same set of expert parameters.'], size=26)
    arrow(d, (390, 475), (390, 530))
    arrow(d, (2010, 475), (2010, 530))
    box(d, (70, 540, 890, 730), 'Full replica: canonical owner 0', ['Gather E3 from GPU3; retain E0 E1 E2.', 'Owner layout: [E0, E1, E2, E3]', 'Key = (layer, parameter slot, expert ID)'], size=27)
    box(d, (1510, 540, 2330, 730), 'Reduced replica: canonical owner 0', ['E0 E1 E2 E3 are already local.', 'Owner layout: [E0, E1, E2, E3]', 'Same keys, same synchronization layout.'], size=27)
    arrow(d, (905, 625), (1495, 625), GREEN, both=True)
    label(d, (965, 565), 'Native owner EDP', 30, GREEN, True)
    label(d, (970, 655), 'E3 syncs with E3', 29, GREEN)
    label(d, (70, 770), 'The reverse map returns synchronized gradients or updated parameters to the correct physical expert.', 29, MUTED)
    d.line((70, 830, 2330, 830), fill='#a7b3ba', width=2)
    label(d, (70, 860), '2. Nondivisible placement: unequal real experts, equal-width fused-dispatch slots', 34, bold=True)
    box(d, (70, 940, 610, 1155), 'Router remains a 10-expert router', ['Scores/routes: E0 ... E9', 'EP3 real counts: 4, 3, 3', 'No E10 or E11 in the model.'], size=26)
    arrow(d, (625, 1045), (740, 1045))
    label(d, (655, 985), 'Map', 25, BLUE)
    box(d, (755, 930, 1615, 1180), 'Replica-local backend slot map: width 4', ['GPU0: [ E0   E1   E2   E3 ]', 'GPU1: [ E4   E5   E6   EMPTY ]', 'GPU2: [ E7   E8   E9   EMPTY ]', '12 virtual slots; exactly 10 real experts.'], size=27)
    arrow(d, (1630, 1045), (1740, 1045))
    box(d, (1755, 940, 2330, 1155), 'Fused Flex / HybridEP dispatch', ['Fixed-width backend metadata.', 'EMPTY routes are masked out.', 'Only real local experts execute.'], True, 26)
    box(d, (70, 1230, 1160, 1430), 'Example token: router selects E9', ['Slot map: E9 -> GPU2, local slot 2.', 'Fused dispatch sends token to the correct real expert.', 'Combine restores the token output; model semantics stay fixed.'], size=27)
    box(d, (1220, 1230, 2330, 1430), 'What is padded, and what is not', ['Padded: routing/transfer layout slots (plus backend alignment).', 'Not padded: trainable experts or expert optimizer state.', 'Healthy and reduced replicas both contain the same 10 experts.'], size=27)
    label(d, (70, 1480), 'NEP additions: explicit global-expert mapping + nonuniform real placement with a uniform virtual interface.', 29, bold=True)
    label(d, (70, 1540), 'Inherited from NTP: the common-owner reshard/synchronize principle. DistOpt integration is shown separately.', 28, MUTED)
    save(im, 'nep_expert_contributions')


if __name__ == '__main__':
    verify_layout()
    systems()
    implementation()
    expert_contributions()
    print(f'Placement invariants verified; figures written to {OUT}')
