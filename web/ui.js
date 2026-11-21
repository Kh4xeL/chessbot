/*
 * UI helpers shared by play and analysis mode: eval graph, accuracy, PGN export
 * (with variations), engine-line rendering and small formatting utilities.
 */

var UI = (function () {
    // ---------------------------------------------------------------- formatting
    function fmtEval(evalPawns, mate) {
        if (mate !== null && mate !== undefined) return (mate > 0 ? 'M' : '-M') + Math.abs(mate);
        if (evalPawns === null || evalPawns === undefined || isNaN(evalPawns)) return '…';
        var v = Number(evalPawns);
        return (v > 0 ? '+' : '') + v.toFixed(2);
    }

    // White-view eval -> CSS colours for a small badge (white-ish when White is better).
    function evalBadge(evalPawns, mate) {
        var whiteBetter = (mate !== null && mate !== undefined) ? mate > 0 : Number(evalPawns) >= 0;
        var bg = whiteBetter ? '#e8e8e8' : '#1b1b1b';
        var fg = whiteBetter ? '#111' : '#eee';
        return '<span class="eval-badge" style="background:' + bg + ';color:' + fg + ';">' +
            fmtEval(evalPawns, mate) + '</span>';
    }

    // Numbered SAN line ("12... Nxd4 13. Bxd4 c5") starting from a FEN.
    function sanLine(fen, sanMoves, maxMoves) {
        var parts = fen.split(' ');
        var white = parts[1] === 'w';
        var num = parseInt(parts[5] || '1', 10);
        var out = [];
        (sanMoves || []).slice(0, maxMoves || 99).forEach(function (san, i) {
            if (white) out.push(num + '. ' + san);
            else out.push((i === 0 ? num + '... ' : '') + san);
            if (!white) num++;
            white = !white;
        });
        return out.join(' ');
    }

    // UCI moves -> SAN from a FEN (chess.js), stops at the first illegal move.
    function uciToSan(fen, uciMoves) {
        var g = new Chess(fen);
        var out = [];
        for (var i = 0; i < uciMoves.length; i++) {
            var u = uciMoves[i];
            var mv = g.move({ from: u.substring(0, 2), to: u.substring(2, 4), promotion: u[4] || 'q' });
            if (!mv) break;
            out.push(mv.san);
        }
        return out;
    }

    // ---------------------------------------------------------------- accuracy
    // Lichess-style: win% from centipawns, accuracy from the win% a move gives away.
    function winPct(cp) { return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * cp)) - 1); }

    function evalCp(m) {
        if (m.mate !== null && m.mate !== undefined) return m.mate > 0 ? 1000 : -1000;
        if (m.is_checkmate) return m.color === 'White' ? 1000 : -1000;
        return Math.max(-1000, Math.min(1000, Math.round((m.eval_pawns || 0) * 100)));
    }

    // moves: main-line analysis moves in order; startCp: eval before the first move.
    function accuracy(moves, startCp) {
        var prev = startCp || 0;
        var acc = { White: [], Black: [] };
        moves.forEach(function (m) {
            var cur = evalCp(m);
            var before = winPct(prev), after = winPct(cur);
            var lost = m.color === 'White' ? before - after : after - before;
            var a = 103.1668 * Math.exp(-0.04354 * Math.max(0, lost)) - 3.1669;
            acc[m.color].push(Math.max(0, Math.min(100, a)));
            prev = cur;
        });
        function mean(xs) { return xs.length ? xs.reduce(function (s, x) { return s + x; }, 0) / xs.length : null; }
        return { White: mean(acc.White), Black: mean(acc.Black) };
    }

    // ---------------------------------------------------------------- eval graph
    // points: [{evalPawns, mate, cls}] in move order; onPick(i) when clicked.
    function renderEvalGraph(container, points, currentIndex, onPick, startEval) {
        var el = typeof container === 'string' ? document.querySelector(container) : container;
        if (!el) return;
        var W = el.clientWidth || 400, H = el.clientHeight || 90;
        var all = [{ evalPawns: startEval || 0, mate: null }].concat(points);
        var n = all.length;
        function y(p) {
            var cp = (p.mate !== null && p.mate !== undefined) ? (p.mate > 0 ? 1000 : -1000)
                : Math.max(-1000, Math.min(1000, (p.evalPawns || 0) * 100));
            return H - (winPct(cp) / 100) * H;
        }
        function x(i) { return n <= 1 ? 0 : (i / (n - 1)) * W; }
        var path = 'M0,' + H;
        for (var i = 0; i < n; i++) path += ' L' + x(i).toFixed(1) + ',' + y(all[i]).toFixed(1);
        path += ' L' + W + ',' + H + ' Z';

        var dots = '';
        points.forEach(function (p, i) {
            var c = p.cls === 'Blunder' ? '#e74c3c' : p.cls === 'Mistake' ? '#e67e22' :
                p.cls === 'Inaccuracy' ? '#f1c40f' : (p.cls === 'Brilliant' || p.cls === 'Perfect') ? '#9b59b6' : null;
            if (c) dots += '<circle cx="' + x(i + 1).toFixed(1) + '" cy="' + y(p).toFixed(1) + '" r="3.5" fill="' + c + '" stroke="#111" stroke-width="1"/>';
        });
        var cur = (currentIndex !== null && currentIndex !== undefined && currentIndex >= -1)
            ? '<line x1="' + x(currentIndex + 1).toFixed(1) + '" y1="0" x2="' + x(currentIndex + 1).toFixed(1) +
              '" y2="' + H + '" stroke="#4CAF50" stroke-width="2"/>' : '';
        el.innerHTML =
            '<svg width="' + W + '" height="' + H + '" style="display:block;cursor:pointer;">' +
            '<rect width="' + W + '" height="' + H + '" fill="#262626"/>' +
            '<path d="' + path + '" fill="#d9d9d9"/>' +
            '<line x1="0" y1="' + H / 2 + '" x2="' + W + '" y2="' + H / 2 + '" stroke="#777" stroke-dasharray="3,3"/>' +
            dots + cur + '</svg>';
        if (onPick) {
            el.querySelector('svg').onclick = function (e) {
                var r = this.getBoundingClientRect();
                var i = Math.round(((e.clientX - r.left) / W) * (n - 1)) - 1;
                onPick(Math.max(-1, Math.min(points.length - 1, i)));
            };
        }
    }

    // ---------------------------------------------------------------- PGN export
    function pgnHeaders(h) {
        return Object.keys(h).filter(function (k) { return h[k] !== undefined && h[k] !== null && h[k] !== ''; })
            .map(function (k) { return '[' + k + ' "' + String(h[k]).replace(/"/g, "'") + '"]'; }).join('\n');
    }

    // Analysis moves form a tree via parentId; the continuation that stays on the
    // parent's branch is the main line, the others become (variations).
    function pgnFromTree(moves, headers, result) {
        var kids = {};
        moves.forEach(function (m) {
            var k = m.parentId || 'root';
            (kids[k] = kids[k] || []).push(m);
        });
        function fmt(m, force) {
            if (m.color === 'White') return m.move_number + '. ' + m.san;
            return (force ? m.move_number + '... ' : '') + m.san;
        }
        function line(parentId, branch, forceNumber) {
            var list = kids[parentId || 'root'] || [];
            if (!list.length) return '';
            var main = list.filter(function (k) { return k.branchId === branch; })[0] || list[0];
            var vars = list.filter(function (k) { return k !== main; });
            var out = fmt(main, forceNumber);
            vars.forEach(function (v) {
                var rest = line(v.id, v.branchId, false);
                out += ' (' + fmt(v, true) + (rest ? ' ' + rest : '') + ')';
            });
            var next = line(main.id, main.branchId, vars.length > 0);
            return out + (next ? ' ' + next : '');
        }
        var body = line(null, 'main', true);
        var res = result || '*';
        return pgnHeaders(headers) + '\n\n' + (body ? body + ' ' : '') + res + '\n';
    }

    function download(filename, text) {
        var blob = new Blob([text], { type: 'application/x-chess-pgn' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
    }

    function copy(text, btn) {
        var done = function () {
            if (!btn) return;
            var old = btn.textContent;
            btn.textContent = '✓ Copied';
            setTimeout(function () { btn.textContent = old; }, 1200);
        };
        if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done);
        else {
            var ta = document.createElement('textarea');
            ta.value = text; document.body.appendChild(ta); ta.select();
            try { document.execCommand('copy'); } catch (e) {}
            ta.remove(); done();
        }
    }

    function today() {
        var d = new Date();
        return d.getFullYear() + '.' + String(d.getMonth() + 1).padStart(2, '0') + '.' + String(d.getDate()).padStart(2, '0');
    }

    // ---------------------------------------------------------------- material
    var VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    var GLYPH = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛' };
    // Captured pieces for each side plus the material balance, from a FEN.
    function material(fen) {
        var start = { p: 8, n: 2, b: 2, r: 2, q: 1 };
        var cnt = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
        fen.split(' ')[0].split('').forEach(function (ch) {
            var lower = ch.toLowerCase();
            if (cnt.w[lower] === undefined) return;
            cnt[ch === lower ? 'b' : 'w'][lower]++;
        });
        function captured(side) {   // pieces of `side` that are gone (promotions can make this negative)
            var s = '';
            ['q', 'r', 'b', 'n', 'p'].forEach(function (t) {
                for (var i = 0; i < start[t] - cnt[side][t]; i++) s += GLYPH[t];
            });
            return s;
        }
        var score = 0;
        Object.keys(VALUES).forEach(function (t) { if (t !== 'k') score += VALUES[t] * (cnt.w[t] - cnt.b[t]); });
        return { whiteLost: captured('w'), blackLost: captured('b'), balance: score };
    }

    return {
        fmtEval: fmtEval, evalBadge: evalBadge, sanLine: sanLine, uciToSan: uciToSan,
        accuracy: accuracy, renderEvalGraph: renderEvalGraph, pgnFromTree: pgnFromTree,
        pgnHeaders: pgnHeaders, download: download, copy: copy, today: today, material: material
    };
})();
