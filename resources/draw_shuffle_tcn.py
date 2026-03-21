"""
Visualization of Shuffle-TCN architecture for publication.

Shuffle-TCN extends MS-TCN (Multi-Scale Temporal Convolutional Network) by
inserting a Channel Shuffle operation (inspired by ShuffleNet, CVPR 2018)
between the feature transformation and batch normalization steps, enabling
cross-branch feature interaction without additional parameters.

Reference: Zhang et al., "ShuffleNet: An Extremely Efficient Convolutional
Neural Network for Mobile Applications", CVPR 2018 (arXiv:1801.07455)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT   = '#2E6DA4'
C_CONV1x1 = '#E8810A'
C_TCN     = '#C0392B'
C_MAXPOOL = '#16A085'
C_CONCAT  = '#27AE60'
C_TRANS   = '#D4AC0D'
C_SHUFFLE = '#8E44AD'
C_BN      = '#C0577A'
C_DROP    = '#7F6000'
C_OUTPUT  = '#2E6DA4'
C_ARROW   = '#2c3e50'
ALPHA     = 0.90


# ─── Helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, color, label, fs=8, bold=False,
         tc='white', alpha=ALPHA, zorder=3):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.025',
                       linewidth=1.3, edgecolor='#1a1a1a',
                       facecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold' if bold else 'normal',
            zorder=zorder+1)

def arr(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12),
                zorder=zorder)


# =============================================================================
# FIGURE LAYOUT
#   Row 0 (top, tall):  Panel A (left 55%) | Panel B (right 45%)
#   Row 1 (bottom):     Panel C (full width)
# =============================================================================
fig = plt.figure(figsize=(20, 22), facecolor='white')

# ── suptitle ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.985,
         'Shuffle-TCN: Multi-Scale Temporal Convolution with Channel Shuffle',
         ha='center', va='top', fontsize=15, fontweight='bold', color='#111111')
fig.text(0.5, 0.970,
         'Channel Shuffle (ShuffleNet, Zhang et al. CVPR 2018, arXiv:1801.07455) '
         'inserted after Transform — zero extra parameters, cross-branch feature interaction.',
         ha='center', va='top', fontsize=9.5, color='#444444', style='italic')

# ── Axes ──────────────────────────────────────────────────────────────────────
# [left, bottom, width, height]  all in figure fraction
ax_A = fig.add_axes([0.01, 0.38, 0.53, 0.58])   # Panel A – full pipeline
ax_B = fig.add_axes([0.56, 0.38, 0.43, 0.58])   # Panel B – channel shuffle
ax_C = fig.add_axes([0.01, 0.01, 0.98, 0.35])   # Panel C – comparison

for ax in [ax_A, ax_B, ax_C]:
    ax.axis('off')
    ax.set_facecolor('#f9f9f9')
    # subtle background
    for spine in ax.spines.values():
        spine.set_visible(False)


# =============================================================================
# PANEL A  – Full Shuffle-TCN pipeline
# =============================================================================
ax_A.set_xlim(0, 12)
ax_A.set_ylim(0, 22)

ax_A.text(6, 21.5, '(A)  Shuffle-TCN Architecture',
          ha='center', va='center', fontsize=12, fontweight='bold',
          color='#1a1a1a')

# ── Input ─────────────────────────────────────────────────────────────────────
rbox(ax_A, 6, 20.6, 3.2, 0.65, C_INPUT,
     'Input  (N, C, T, V)', fs=9.5, bold=True)
arr(ax_A, 6, 20.27, 6, 19.85)

# ── Split label ──────────────────────────────────────────────────────────────
ax_A.axhline(19.85, xmin=0.04, xmax=0.96,
             color='#bbbbbb', lw=0.9, linestyle='--')
ax_A.text(6, 19.92, 'Multi-scale branch split',
          ha='center', va='bottom', fontsize=7.5, color='#888888', style='italic')

# ── Branch definitions ────────────────────────────────────────────────────────
xs_br  = [1.2, 3.1, 5.0, 6.9, 8.8, 10.7]
n_br   = 6
BRW    = 1.55   # branch block width
BRH    = 0.50   # branch block height

branch_defs = [
    # (top_label,   bottom_label,   top_color,   bot_color)
    ('Conv 1×1\nBN + ReLU',  'TCN k=3, d=1',  C_CONV1x1, C_TCN),
    ('Conv 1×1\nBN + ReLU',  'TCN k=3, d=2',  C_CONV1x1, C_TCN),
    ('Conv 1×1\nBN + ReLU',  'TCN k=3, d=3',  C_CONV1x1, C_TCN),
    ('Conv 1×1\nBN + ReLU',  'TCN k=3, d=4',  C_CONV1x1, C_TCN),
    ('Conv 1×1\nBN + ReLU',  'MaxPool k=3',   C_CONV1x1, C_MAXPOOL),
    ('Conv 1×1 (stride)',     '',              C_CONV1x1, None),
]

BRANCH_TOP_Y = 19.7   # starting y for branch fan-out arrows
BR_ROW1_Y   = 18.85  # first row of blocks centre
BR_ROW2_Y   = 17.75  # second row

branch_bot_ys = []   # bottom edge of each branch's last block
for i, (xi, (lbl1, lbl2, c1, c2)) in enumerate(zip(xs_br, branch_defs)):
    # fan-out arrow
    arr(ax_A, 6, BRANCH_TOP_Y, xi, BR_ROW1_Y + BRH/2 + 0.05,
        color='#aaaaaa', lw=0.9)
    # top block (Conv1×1 + BN/ReLU)
    rbox(ax_A, xi, BR_ROW1_Y, BRW, BRH + 0.18, c1, lbl1, fs=7)
    if lbl2:
        arr(ax_A, xi, BR_ROW1_Y - (BRH + 0.18)/2,
            xi, BR_ROW2_Y + BRH/2 + 0.05, color='#aaaaaa', lw=0.9)
        rbox(ax_A, xi, BR_ROW2_Y, BRW, BRH, c2, lbl2, fs=7.5, bold=True)
        branch_bot_ys.append(BR_ROW2_Y - BRH/2)
    else:
        branch_bot_ys.append(BR_ROW1_Y - (BRH + 0.18)/2)

# ── Merge label ───────────────────────────────────────────────────────────────
CAT_Y = 16.65
ax_A.axhline(CAT_Y + 0.55, xmin=0.04, xmax=0.96,
             color='#bbbbbb', lw=0.9, linestyle='--')
ax_A.text(6, CAT_Y + 0.62, 'Branch merge',
          ha='center', va='bottom', fontsize=7.5, color='#888888', style='italic')

for xi, bot in zip(xs_br, branch_bot_ys):
    arr(ax_A, xi, bot, 6, CAT_Y + 0.32, color='#aaaaaa', lw=0.9)

# ── Concat ────────────────────────────────────────────────────────────────────
rbox(ax_A, 6, CAT_Y, 4.0, 0.65, C_CONCAT,
     'Concatenate  (channel dim)', fs=9, bold=True)
arr(ax_A, 6, CAT_Y - 0.32, 6, 15.62)

# ── Transform block ───────────────────────────────────────────────────────────
TRANS_CY = 15.35
rbox(ax_A, 6, TRANS_CY + 0.35, 3.6, 0.52, C_BN, 'BN + ReLU', fs=8.5)
arr(ax_A, 6, TRANS_CY + 0.35 - 0.26, 6, TRANS_CY - 0.35 + 0.26)
rbox(ax_A, 6, TRANS_CY - 0.35, 3.6, 0.52, C_CONV1x1,
     'Conv 1×1  →  C_out channels', fs=8.5)

# bracket around transform
bx_l = 6 - 3.6/2 - 0.25
bx_w = 3.6 + 0.5
by_b = TRANS_CY - 0.35 - 0.28
by_h = 1.48
ax_A.add_patch(FancyBboxPatch((bx_l, by_b), bx_w, by_h,
               boxstyle='round,pad=0.06',
               linewidth=1.8, edgecolor=C_TRANS, facecolor='none', zorder=2))
ax_A.text(bx_l + bx_w + 0.08, by_b + by_h/2,
          'Transform\nBlock', fontsize=8, color=C_TRANS,
          ha='left', va='center', fontweight='bold')

arr(ax_A, 6, TRANS_CY - 0.35 - 0.26, 6, 13.62)

# ── Channel Shuffle  ★ ────────────────────────────────────────────────────────
SH_Y = 13.32
rbox(ax_A, 6, SH_Y, 4.2, 0.72, C_SHUFFLE,
     'Channel Shuffle  (g groups)', fs=11, bold=True, alpha=0.93)
# star marker
ax_A.text(6 + 4.2/2 + 0.22, SH_Y,
          '★  Novel', fontsize=9, color=C_SHUFFLE,
          ha='left', va='center', fontweight='bold')

arr(ax_A, 6, SH_Y - 0.36, 6, 12.32)

# ── BN ────────────────────────────────────────────────────────────────────────
BN2_Y = 12.05
rbox(ax_A, 6, BN2_Y, 3.2, 0.58, C_BN, 'BatchNorm2d', fs=9)
arr(ax_A, 6, BN2_Y - 0.29, 6, 11.10)

# ── Dropout ───────────────────────────────────────────────────────────────────
DROP_Y = 10.82
rbox(ax_A, 6, DROP_Y, 3.2, 0.58, C_DROP, 'Dropout', fs=9)
arr(ax_A, 6, DROP_Y - 0.29, 6, 9.85)

# ── Output ────────────────────────────────────────────────────────────────────
OUT_Y = 9.58
rbox(ax_A, 6, OUT_Y, 3.6, 0.65, C_OUTPUT,
     'Output  (N, C_out, T, V)', fs=9.5, bold=True)

# ── Step index labels (left margin) ──────────────────────────────────────────
steps = [
    (20.6,  '① Input'),
    (18.3,  '② Multi-scale\n    Branches'),
    (CAT_Y, '③ Concatenate'),
    (TRANS_CY, '④ Transform'),
    (SH_Y,  '⑤ Channel\n    Shuffle'),
    (BN2_Y, '⑥ BN'),
    (DROP_Y,'⑦ Dropout'),
    (OUT_Y, '⑧ Output'),
]
for sy, sl in steps:
    ax_A.text(0.05, sy, sl, ha='left', va='center',
              fontsize=7.5, color='#555555', style='italic')


# =============================================================================
# PANEL B  – Channel Shuffle mechanism
# =============================================================================
ax_B.set_xlim(0, 10)
ax_B.set_ylim(0, 22)

ax_B.text(5, 21.5, '(B)  Channel Shuffle Mechanism',
          ha='center', va='center', fontsize=12, fontweight='bold',
          color='#1a1a1a')
ax_B.text(5, 20.85,
          'Inspired by ShuffleNet (Zhang et al., CVPR 2018)',
          ha='center', va='center', fontsize=9, color='#555555', style='italic')

# ── Draw coloured channel strips ──────────────────────────────────────────────
N_GRP   = 2
N_CH    = 4     # channels per group
STRIP_W = 0.75
STRIP_H = 4.5
GAP     = 0.10
TOTAL   = N_GRP * N_CH
x0 = 5 - (TOTAL * (STRIP_W + GAP)) / 2 + GAP/2

GRP_COLORS = [
    ['#2E6DA4', '#4A85BC', '#6699CC', '#88AEDD'],   # blues
    ['#C0392B', '#D4564A', '#E87367', '#F59084'],   # reds
]

def draw_strips(ax, left_x, top_y, colors, label):
    """Draw N coloured strips and label them."""
    for i, c in enumerate(colors):
        xi = left_x + i * (STRIP_W + GAP)
        p = FancyBboxPatch((xi, top_y - STRIP_H), STRIP_W, STRIP_H,
                           boxstyle='round,pad=0.02',
                           facecolor=c, edgecolor='white', lw=0.8,
                           alpha=0.92, zorder=3)
        ax.add_patch(p)
        ax.text(xi + STRIP_W/2, top_y - STRIP_H/2,
                str(i+1), ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)
    ax.text(left_x + TOTAL*(STRIP_W+GAP)/2 - GAP/2,
            top_y - STRIP_H - 0.25, label,
            ha='center', va='top', fontsize=8.5, color='#333333')

# ── BEFORE ───────────────────────────────────────────────────────────────────
BEFORE_TOP = 19.8
ax_B.text(5, BEFORE_TOP + 0.3, 'Before  (channels ordered by group)',
          ha='center', va='bottom', fontsize=9.5, fontweight='bold',
          color='#333333')

flat_before = GRP_COLORS[0] + GRP_COLORS[1]
draw_strips(ax_B, x0, BEFORE_TOP, flat_before, '')

# Group brackets
for g in range(N_GRP):
    gx = x0 + g * N_CH * (STRIP_W + GAP)
    gw = N_CH * (STRIP_W + GAP) - GAP
    ax_B.annotate('',
                  xy=(gx + gw, BEFORE_TOP + 0.22),
                  xytext=(gx, BEFORE_TOP + 0.22),
                  arrowprops=dict(arrowstyle='|-|', color=GRP_COLORS[g][0],
                                  lw=1.8, mutation_scale=5),
                  zorder=4)
    ax_B.text(gx + gw/2, BEFORE_TOP + 0.35,
              f'Group {g}', ha='center', va='bottom', fontsize=9,
              color=GRP_COLORS[g][0], fontweight='bold')

# Text description
ax_B.text(5, BEFORE_TOP - STRIP_H - 0.55,
          '[ G₀C₀  G₀C₁  G₀C₂  G₀C₃ | G₁C₀  G₁C₁  G₁C₂  G₁C₃ ]',
          ha='center', va='top', fontsize=9, color='#333333',
          fontfamily='monospace')

# ── Arrow + reshape labels ────────────────────────────────────────────────────
MID_Y = BEFORE_TOP - STRIP_H - 2.8
ax_B.annotate('', xy=(5, MID_Y + 1.35), xytext=(5, MID_Y + 2.2),
              arrowprops=dict(arrowstyle='->', color=C_SHUFFLE, lw=2.5,
                              mutation_scale=18), zorder=4)

steps_txt = [
    '① Reshape:  (N,C,T,V) → (N, g, C/g, T, V)',
    '② Transpose: dim 1 ↔ dim 2  →  (N, C/g, g, T, V)',
    '③ Reshape:  (N, C/g, g, T, V) → (N, C, T, V)',
]
for k, s in enumerate(steps_txt):
    ax_B.text(5, MID_Y + 1.1 - k*0.48, s,
              ha='center', va='top', fontsize=8.5, color='#333333',
              style='italic')

# ── AFTER ─────────────────────────────────────────────────────────────────────
AFTER_TOP = MID_Y - 0.05
ax_B.text(5, AFTER_TOP + 0.28, 'After  (channels interleaved across groups)',
          ha='center', va='bottom', fontsize=9.5, fontweight='bold',
          color='#333333')

interleaved = [GRP_COLORS[g][c]
               for c in range(N_CH) for g in range(N_GRP)]
draw_strips(ax_B, x0, AFTER_TOP, interleaved, '')

ax_B.text(5, AFTER_TOP - STRIP_H - 0.55,
          '[ G₀C₀  G₁C₀  G₀C₁  G₁C₁  G₀C₂  G₁C₂  G₀C₃  G₁C₃ ]',
          ha='center', va='top', fontsize=9, color='#333333',
          fontfamily='monospace')

# ── Key property box ──────────────────────────────────────────────────────────
ax_B.text(5, AFTER_TOP - STRIP_H - 2.05,
          'Key Property:\nChannel Shuffle enables cross-branch feature\n'
          'interaction with ZERO learnable parameters.\n'
          'Pure reshape + transpose — no convolution needed.',
          ha='center', va='top', fontsize=9, color='#333333',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0ecf8',
                    edgecolor=C_SHUFFLE, lw=1.5, alpha=0.9))


# =============================================================================
# PANEL C  – MS-TCN vs Shuffle-TCN comparison
# =============================================================================
ax_C.set_xlim(0, 20)
ax_C.set_ylim(0, 8)

ax_C.text(10, 7.72, '(C)  Architecture Comparison: MS-TCN (Baseline)  vs  Shuffle-TCN (Ours)',
          ha='center', va='center', fontsize=12, fontweight='bold',
          color='#1a1a1a')

# ── Two pipeline columns ──────────────────────────────────────────────────────
shared = [
    ('Input  (N, C, T, V)',           C_INPUT),
    ('Multi-scale Branches',           C_TCN),
    ('Concatenate',                    C_CONCAT),
    ('Transform  (BN → ReLU → Conv 1×1)', C_TRANS),
]
mstcn_extra   = [('BatchNorm2d', C_BN)]
shuffle_extra = [('Channel Shuffle  ★', C_SHUFFLE), ('BatchNorm2d', C_BN)]
tail = [('Dropout', C_DROP), ('Output  (N, C, T, V)', C_OUTPUT)]

def draw_col(ax, cx, title, tc, blocks):
    ax.text(cx, 7.35, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color=tc)
    step = 0.90
    top_y = 6.85
    prev_y = None
    bw = 5.5
    for i, (lbl, col) in enumerate(blocks):
        y = top_y - i * step
        p = FancyBboxPatch((cx - bw/2, y - 0.31), bw, 0.62,
                           boxstyle='round,pad=0.025',
                           linewidth=1.3, edgecolor='#1a1a1a',
                           facecolor=col, alpha=ALPHA, zorder=3)
        ax.add_patch(p)
        ax.text(cx, y, lbl, ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold', zorder=4)
        if prev_y is not None:
            ax.annotate('', xy=(cx, y+0.31), xytext=(cx, prev_y-0.31),
                        arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.4),
                        zorder=2)
        prev_y = y
    return prev_y

draw_col(ax_C, 5,  'MS-TCN  (baseline)', C_TCN,
         shared + mstcn_extra + tail)
r = draw_col(ax_C, 15, 'Shuffle-TCN  (ours)', C_SHUFFLE,
             shared + shuffle_extra + tail)

# ── Insertion annotation ──────────────────────────────────────────────────────
# The Channel Shuffle sits between Transform and BN → at index 4 in shuffle col
idx_sh = len(shared)               # 4th block (0-indexed)
sh_y   = 6.85 - idx_sh * 0.90
ax_C.annotate('',
              xy=(15 - 5.5/2 - 0.1, sh_y),
              xytext=(10.8, sh_y),
              arrowprops=dict(arrowstyle='->', color=C_SHUFFLE,
                              lw=2.0, mutation_scale=14),
              zorder=4)
ax_C.text(10.6, sh_y, '+ Channel\n   Shuffle\n   inserted',
          ha='right', va='center', fontsize=8, color=C_SHUFFLE,
          fontweight='bold')

# ── Vertical divider ──────────────────────────────────────────────────────────
ax_C.axvline(10, ymin=0.05, ymax=0.95,
             color='#cccccc', lw=1.2, linestyle='--', zorder=1)

# ── Summary note ──────────────────────────────────────────────────────────────
ax_C.text(10, 0.45,
          'Shuffle-TCN adds ONE operation (Channel Shuffle) with ZERO extra parameters.\n'
          'Cross-branch feature interaction is achieved by a pure reshape + transpose, '
          'promoting information flow between parallel temporal receptive fields.',
          ha='center', va='center', fontsize=9, color='#333333',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0ecf8',
                    edgecolor=C_SHUFFLE, lw=1.3, alpha=0.9))


# =============================================================================
# Legend
# =============================================================================
legend_items = [
    (C_INPUT,   'Input / Output tensor'),
    (C_CONV1x1, '1×1 Convolution'),
    (C_TCN,     'Dilated Temporal Conv'),
    (C_MAXPOOL, 'Max-Pool branch'),
    (C_CONCAT,  'Concatenation'),
    (C_TRANS,   'Transform block'),
    (C_SHUFFLE, 'Channel Shuffle  ★'),
    (C_BN,      'Batch Normalisation'),
    (C_DROP,    'Dropout'),
]
handles = [mpatches.Patch(facecolor=c, edgecolor='#555', label=l)
           for c, l in legend_items]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.92,
           bbox_to_anchor=(0.5, -0.002),
           edgecolor='#aaaaaa', title='Component Legend',
           title_fontsize=9)


# =============================================================================
out_path = ('/home/theo2204/python-workspace/openmmlab-repo/'
            'mmaction2_shuffle_stgcn/resources/shuffle_tcn_architecture.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved → {out_path}')
