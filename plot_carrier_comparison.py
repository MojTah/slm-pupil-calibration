"""Plot the verified matched-grid command comparison; numerical inputs are local."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--curves', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    assert summary['status'] == 'PASS'
    assert hashlib.sha256(args.curves.read_bytes()).hexdigest() == summary['output_sha256']['phase-curves.csv']
    with args.curves.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert set(r['command'] for r in rows) == {'original', 'orthogonal'}
    records = {r['command']: r for r in summary['results'] if r['domain_D'] == 6}
    assert all(r['modulation_angles'] == 256 and r['readout'] == 'archived_fixed' for r in records.values())
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'normal'})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.5), sharex=True)
    gain_limits = (min(r['e_B_min_percent'] for r in records.values()),
                   max(r['e_B_max_percent'] for r in records.values()))
    gain_margin = .08 * (gain_limits[1] - gain_limits[0])
    for col, (command, title, color) in enumerate([
            ('original', r'Original strip carrier $(-q,0)$', '#b46024'),
            ('orthogonal', r'Orthogonal strip carrier $(0,+q)$', '#24678d')]):
        data = [r for r in rows if r['command'] == command]
        phase, error, offset = (np.array([float(r[k]) for r in data])
                                for k in ('phase_rad', 'e_B_percent', 'zero_offset_vs_B_rad'))
        assert len(phase) == 721 and np.all(np.isfinite([phase, error, offset]))
        assert np.all(np.diff(phase) > 0) and np.isclose(phase[-1], 2*np.pi)
        record = records[command]
        assert error.min() >= record['e_B_min_percent']-1e-10
        assert error.max() <= record['e_B_max_percent']+1e-10
        axes[0, col].plot(phase, error, color=color, lw=1.8, label='Coherent')
        axes[0, col].axhline(record['e_B_pair_percent'], color='#333333', ls='--', lw=1.2,
                            label='Ideal raw pair')
        axes[0, col].set_title(title, fontsize=9.5, pad=18)
        axes[0, col].set_ylabel(r'Scale error $e_B$ (%)')
        axes[0, col].set_ylim(gain_limits[0]-gain_margin, gain_limits[1]+gain_margin)
        axes[0, col].legend(frameon=False, fontsize=8, loc='lower center')
        if command == 'orthogonal':
            zoom = axes[0, col].inset_axes([.26, .58, .66, .31])
            zoom.plot(phase, error, color=color, lw=1.4)
            zoom.axhline(record['e_B_pair_percent'], color='#333333', ls='--', lw=1)
            zoom.set_title('Orthogonal zoom (%)', fontsize=7, pad=3)
            zoom.set_xlim(0, 2*np.pi)
            zoom.set_xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'$2\pi$'])
            zoom.set_yticks([-.075, -.065, -.055])
            zoom.tick_params(labelsize=6.5, pad=1)
            zoom.grid(alpha=.17, lw=.6)
            zoom.ticklabel_format(axis='y', style='plain', useOffset=False)
        axes[1, col].plot(phase, offset, color=color, lw=1.8)
        axes[1, col].set_ylabel(r'Raw zero offset $b_B$ (rad)')
        axes[1, col].set_xlabel(r'Strip phase $\phi$')
        axes[1, col].text(.5, 1.04, f"Span {record['zero_offset_span_rad']:.4g} rad",
                          transform=axes[1, col].transAxes, ha='center', fontsize=8)
        for row in (0, 1):
            ax = axes[row, col]
            ax.set_xlim(0, 2*np.pi)
            ax.set_xticks(np.arange(5)*np.pi/2, ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
            ax.grid(alpha=.17, lw=.6)
            ax.text(0, 1.02, f"({chr(97+2*row+col)})", transform=ax.transAxes, fontsize=9)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.subplots_adjust(left=.11, right=.97, bottom=.11, top=.82, wspace=.38, hspace=.42)
    fig.text(.55, .96, '6D domain · 256 modulation angles · fixed archived readout', ha='center', fontsize=9)
    fig.text(.55, .925, 'Gain panels: common vertical scale; inset: orthogonal zoom', ha='center', fontsize=8, color='#444444')
    fig.text(.55, .895, 'Offset panels: individual vertical scales', ha='center', fontsize=8, color='#444444')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output_dir/'carrier-comparison.pdf')
    fig.savefig(args.output_dir/'carrier-comparison.png', dpi=200)
    plt.close(fig)
    (args.output_dir/'figure-provenance.json').write_text(json.dumps({
        'summary_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        'curves_sha256': hashlib.sha256(args.curves.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows': len(rows), 'domain_D': 6, 'modulation_angles': 256,
        'offset_definition': summary['definitions']['zero_offset']}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
