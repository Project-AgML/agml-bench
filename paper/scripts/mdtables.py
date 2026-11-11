import re, sys
def sections(path):
    out, sec, rows, hdr = {}, None, [], None
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('#'):
            if sec and rows: out.setdefault(sec, []).extend(rows)
            sec, rows, hdr = line.lstrip('# ').strip(), [], None
        elif line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if hdr is None: hdr = cells
            elif set(cells[0]) <= set(':-'): continue
            else: rows.append(dict(zip(hdr, cells)))
        else:
            if rows: out.setdefault(sec, []).extend(rows); rows = []
            hdr = None
    if sec and rows: out.setdefault(sec, []).extend(rows)
    return out
