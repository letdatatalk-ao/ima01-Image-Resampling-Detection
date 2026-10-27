"""Point d'avancement — réimplémentation 1D de la détection de rééchantillonnage.

Projet IMA01 — Aymen Ouerghi & Mohamed Missaoui
D'après : Li et al., « Image Resampling Detection via Spectral Correlation
with False Alarm Control ».

Une présentation guidée par des questions : chaque notion répond à une
question, et sa réponse amène la question suivante. Une barre en bas de
l'écran montre où l'on en est.

Tous les chiffres et courbes viennent de NOTRE code (dossier src/) :
resample_1d, dft, pcc_complex, patch_correlations_1d, count_votes,
compute_nfa..., avec les paramètres et les graines de experiences/*.py.
Le premier lancement calcule tout (~1 min, surtout E4) et l'enregistre dans
animation_cache.json ; supprimer ce fichier après une modification de src/.

À placer dans code_ima/ (à côté de src/), puis :
    manimgx preview avancement_1D.py      # aperçu
    manimgx render  avancement_1D.py      # vidéo MP4
    manimgx present avancement_1D.py      # diaporama : une section = une slide
Dépendances : manimgx, numpy, scipy (pour src/acontrario.py).
"""

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

import manimgx as m

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.acontrario import binomial_tail, compute_nfa, count_votes, decide  # noqa: E402
from src.dft import dft  # noqa: E402
from src.resampling import (  # noqa: E402
    cubic_kernel,
    kernel_nn,
    lanczos_kernel,
    linear_kernel,
    resample_1d,
)
from src.spectral import patch_correlations_1d, pcc_complex  # noqa: E402

CACHE = HERE / "animation_cache.json"

ORIG = m.BLUE
RESAMP = m.ORANGE
OK = m.GREEN
BAD = m.RED
ACCENT = m.YELLOW
DIM = m.GREY_B

CHAPTERS = ["Contexte", "A · Corrélation (B3)", "B · Décision (B4)", "Code", "Résultats", "Suite"]


# ---------------------------------------------------------------------------
# 1. Les résultats, calculés avec notre code
# ---------------------------------------------------------------------------


def log10(x: float) -> float:
    return float(np.log10(x)) if x > 0 else -300.0


def compute_results() -> dict:
    res: dict = {}
    M, N, r, L = 100, 250, 20, 10

    # l'exemple qui sert de fil rouge (test.py) : M=100 → N=250, plus proche voisin
    rng = np.random.default_rng(0)
    u = rng.standard_normal(M)
    v = resample_1d(u, N, kernel_nn)
    w = rng.standard_normal(N)
    U, V, W = dft(u), dft(v), dft(w)
    rho = patch_correlations_1d(V, L, N - 1)
    rho_ref = patch_correlations_1d(W, L, N - 1)
    nfa = compute_nfa(rho, r)
    nfa_ref = compute_nfa(rho_ref, r)
    k = count_votes(rho, r)
    n_patches = rho.shape[0]
    res["demo"] = {
        "u": u[:20].tolist(),
        "v": v[:50].tolist(),
        "absU": np.abs(U).tolist(),
        "absV": np.abs(V).tolist(),
        "pcc_copy": float(pcc_complex(V[40:50], V[140:150])),
        "pcc_other": float(pcc_complex(V[40:50], V[77:87])),
        "rho_patch4": rho[4].tolist(),
        "rho_mean": rho.mean(axis=0).tolist(),
        "rho_ref_mean": rho_ref.mean(axis=0).tolist(),
        "rho4_mean": patch_correlations_1d(V, 4, N - 1).mean(axis=0).tolist(),
        "rho4_ref_mean": patch_correlations_1d(W, 4, N - 1).mean(axis=0).tolist(),
        "k": k.tolist(),
        "k_ref": count_votes(rho_ref, r).tolist(),
        "rho": rho.tolist(),
        "rho_ref": rho_ref.tolist(),
        "n_patches": int(n_patches),
        "p_k100": log10(binomial_tail(k[100], n_patches, 1 / (2 * r + 1))),
        "lognfa": [log10(x) if np.isfinite(x) else None for x in nfa],
        "lognfa_ref": [log10(x) if np.isfinite(x) else None for x in nfa_ref],
        "det": decide(nfa, 1.0).tolist(),
        "det_ref": decide(nfa_ref, 1.0).tolist(),
    }

    # E1 : taille de patch L (exp_L_1D.py)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(M)
    v = resample_1d(u, N, kernel_nn)
    w = rng.standard_normal(N)
    V, W = dft(v), dft(w)
    e1 = {"L": [2, 4, 8, 16, 32, 50], "fond": [], "pic": [], "nfa_pic": [], "nfa_ref": []}
    for LL in e1["L"]:
        rh = patch_correlations_1d(V, LL, N - 1)
        rm = rh.mean(axis=0)
        mask = np.ones(N, dtype=bool)
        mask[0] = False
        mask[95:106] = False
        mask[145:156] = False
        e1["fond"].append(float(np.median(rm[mask])))
        e1["pic"].append(float(rm[100]))
        e1["nfa_pic"].append(log10(compute_nfa(rh, r)[100]))
        e1["nfa_ref"].append(log10(np.min(compute_nfa(patch_correlations_1d(W, LL, N - 1), r))))
    res["E1"] = e1

    # E2 : rayon r (exp_r_1D.py), L=8
    rh = patch_correlations_1d(V, 8, N - 1)
    rh_ref = patch_correlations_1d(W, 8, N - 1)
    e2 = {"r": [2, 5, 10, 20, 40, 80], "k": [], "nfa_pic": [], "nfa_ref": []}
    for rr in e2["r"]:
        e2["k"].append(int(count_votes(rh, rr)[100]))
        e2["nfa_pic"].append(log10(compute_nfa(rh, rr)[100]))
        e2["nfa_ref"].append(log10(np.min(compute_nfa(rh_ref, rr))))
    res["E2"] = e2

    # E3 : noyau (exp_kernel_1D.py), L=8
    rng = np.random.default_rng(0)
    u = rng.standard_normal(M)
    e3 = {}
    for name, ker in (
        ("plus proche voisin", kernel_nn),
        ("linéaire", linear_kernel),
        ("cubique", cubic_kernel),
        ("Lanczos", lanczos_kernel),
    ):
        e3[name] = log10(compute_nfa(patch_correlations_1d(dft(resample_1d(u, N, ker)), 8, N - 1), r)[100])
    res["E3"] = e3

    # E4 : facteur N/M (exp_fact_1D.py), L=8 — même suite de tirages
    rng = np.random.default_rng(0)
    e4 = {"q": [], "nfa": [], "det": []}
    for q in np.arange(0.25, 10.01, 0.25):
        NN = int(round(q * M))
        if NN < 2 * r + 2:
            continue
        uu = rng.standard_normal(M)
        nf = compute_nfa(patch_correlations_1d(dft(resample_1d(uu, NN, kernel_nn)), 8, NN - 1), r)
        e4["q"].append(float(q))
        e4["nfa"].append(log10(np.min(nf)))
        e4["det"].append(bool(len(decide(nf, 1.0)) > 0))
    res["E4"] = e4

    # E5 : r optimal selon d (exp_r_vs_d_1D.py), L=10, noyau linéaire
    rng = np.random.default_rng(0)
    u = rng.standard_normal(M)
    rh = patch_correlations_1d(dft(resample_1d(u, N, linear_kernel)), 10, N - 1)
    r_values = np.arange(2, N // 2, 2)
    logs = np.full((len(r_values), N), np.inf)
    for i, rr in enumerate(r_values):
        with np.errstate(divide="ignore"):
            logs[i] = np.log10(compute_nfa(rh, rr))
    r_opt = []
    for d in range(N):
        col = logs[:, d]
        r_opt.append(float(r_values[int(np.argmin(col))]) if np.isfinite(col).any() else None)
    res["E5"] = {"r_opt": r_opt}
    return res


def load_results() -> dict:
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if "rho_ref" in data.get("demo", {}):
            return data
    res = compute_results()
    CACHE.write_text(json.dumps(res), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# 2. Outils de mise en page
# ---------------------------------------------------------------------------

BOTTOM = -3.3  # le contenu reste au-dessus : la barre de progression est dessous


def title_text(s: str) -> m.Text:
    t = m.Text(s, font_size=34, weight=m.BOLD, color=ACCENT)
    if t.width > 13.2:
        t.scale_to_fit_width(13.2)
    return t.to_edge(m.UP, buff=0.35)


class Chart(m.VGroup):
    """Des axes dont l'axe horizontal est toujours en bas, graduations écrites à la main."""

    def __init__(
        self,
        x_range: tuple[float, float, float],
        y_range: tuple[float, float, float],
        width: float,
        height: float,
        x_label: str,
        y_label: str,
        font_size: int = 24,
        x_numbers: bool = True,
    ) -> None:
        super().__init__()
        self.y0 = y_range[0]
        self.axes = m.Axes(
            x_range=[x_range[0], x_range[1], x_range[2]],
            y_range=[0, y_range[1] - y_range[0], y_range[2]],
            x_length=width,
            y_length=height,
            tips=False,
            axis_config={"color": DIM, "stroke_width": 2},
            x_axis_config={"include_numbers": x_numbers, "font_size": font_size},
        )
        ticks = np.arange(y_range[0], y_range[1] + 1e-9, y_range[2])
        self.y_nums = m.VGroup(
            *[
                m.MathTex(f"{t:g}", font_size=font_size).next_to(self.p(x_range[0], t), m.LEFT, buff=0.22)
                for t in ticks
            ]
        )
        self.x_lab = m.MathTex(x_label, font_size=font_size + 8).next_to(self.axes.x_axis, m.RIGHT, buff=0.25)
        self.y_lab = m.MathTex(y_label, font_size=font_size + 6).rotate(math.pi / 2)
        self.y_lab.next_to(self.y_nums, m.LEFT, buff=0.12)
        self.add(self.axes, self.y_nums, self.x_lab, self.y_lab)

    def p(self, x: float, y: float) -> np.ndarray:
        return self.axes.c2p(x, y - self.y0)

    def line(self, xs, ys, color, width: float = 3, clip: tuple[float, float] | None = None) -> m.VMobject:
        pts = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            if clip:
                y = min(max(y, clip[0]), clip[1])
            pts.append(self.p(x, y))
        mob = m.VMobject(stroke_color=color, stroke_width=width, fill_opacity=0)
        mob.set_points_as_corners(pts)
        return mob

    def stems(self, xs, ys, color, width: float = 3) -> m.VGroup:
        return m.VGroup(*[m.Line(self.p(x, 0), self.p(x, y), color=color, stroke_width=width) for x, y in zip(xs, ys)])

    def dots(self, xs, ys, colors, radius: float = 0.06) -> m.VGroup:
        return m.VGroup(*[m.Dot(self.p(x, y), radius=radius, color=c) for x, y, c in zip(xs, ys, colors)])

    def hline(self, y: float, color=BAD) -> m.DashedLine:
        x0, x1 = self.axes.x_range[0], self.axes.x_range[1]
        return m.DashedLine(self.p(x0, y), self.p(x1, y), color=color, stroke_width=2)


def takeaway_box(text: str, color=OK, font_size: int = 22) -> m.VGroup:
    t = m.Text(text, font_size=font_size, line_spacing=1.15)
    if t.width > 12.4:
        t.scale_to_fit_width(12.4)
    tag = m.Text("À retenir", font_size=16, color=color, weight=m.BOLD)
    box = m.RoundedRectangle(
        width=t.width + 0.6, height=t.height + 0.45, corner_radius=0.12, stroke_color=color, stroke_width=2
    ).set_fill(m.BLACK, 0.85)
    box.move_to(t)
    tag.next_to(box.get_corner(m.UL), m.RIGHT, buff=0.15).shift(0.02 * m.UP)
    tag_bg = m.Rectangle(width=tag.width + 0.16, height=tag.height + 0.08, stroke_width=0).set_fill(m.BLACK, 1)
    tag_bg.move_to(tag)
    g = m.VGroup(box, t, tag_bg, tag)
    g.move_to([0, BOTTOM + g.height / 2 + 0.05, 0])
    return g


# ---------------------------------------------------------------------------
# 3. La scène
# ---------------------------------------------------------------------------




class Avancement1D(m.Scene):
    """Le fil logique suit notre fiche « De la corrélation spectrale au NFA » :
    chaque diapositive répond à une question Qn ; l'encadré « En une phrase »
    est la réponse courte, la visualisation la justifie avec nos résultats."""

    def construct(self) -> None:
        self.R = load_results()
        self.D = self.R["demo"]
        self.cur_title: m.Mobject | None = None
        self.progress = self.make_progress()
        # Contexte
        self.intro()
        self.context_problem()
        self.context_resampling()
        # Partie A — de la corrélation à la matrice ρ (B3)
        self.q1_fourier()
        self.q2_q3_patches()
        self.q4_pcc()
        self.q5_q6_conj_abs()
        self.q7_why_pcc()
        self.q10_key_result()
        # Partie B — la décision a contrario (B4)
        self.q11_no_threshold()
        self.q12_background()
        self.q13_count_votes()
        self.q14_helmholtz()
        self.q15_q16_vote()
        self.q17_binomial()
        self.q18_multiple_tests()
        self.q19_nfa()
        self.q20_decision()
        # Notre code, nos résultats
        self.code_and_tests()
        self.e1_patch_size()
        self.e2_radius()
        self.e3_kernels()
        self.e4_factor()
        self.e5_r_opt()
        # Suite
        self.traps()
        self.open_issues()
        self.q8_q9_towards_2d()
        self.next_steps()

    # -- barre de progression -------------------------------------------------
    def make_progress(self) -> m.VGroup:
        items = m.VGroup(*[m.Text(c, font_size=17, color=m.GREY_D) for c in CHAPTERS]).arrange(m.RIGHT, buff=0.6)
        dots = m.VGroup(*[m.Dot(radius=0.035, color=m.GREY_D) for _ in CHAPTERS[:-1]])
        for i, d in enumerate(dots):
            d.move_to((items[i].get_right() + items[i + 1].get_left()) / 2)
        bar = m.VGroup(items, dots).to_edge(m.DOWN, buff=0.18)
        line = m.Line([-7.1, bar.get_top()[1] + 0.12, 0], [7.1, bar.get_top()[1] + 0.12, 0], stroke_width=1, color=m.GREY_D)
        return m.VGroup(line, bar)

    def set_chapter(self, i: int) -> list[m.Animation]:
        return [
            it.animate.set_color(ACCENT if j == i else (m.GREY_B if j < i else m.GREY_D))
            for j, it in enumerate(self.progress[1][0])
        ]

    # -- une question, qui devient le titre ---------------------------------
    def ask(self, tag: str, question: str, chapter: int, hint: str | None = None) -> m.Text:
        self.next_section(f"{tag} {question}")
        old = [mob for mob in self.mobjects if mob is not self.progress]
        label = m.Text(tag, font_size=24, color=DIM, weight=m.BOLD)
        q = m.Text(question, font_size=42, weight=m.BOLD, color=ACCENT)
        if q.width > 12.8:
            q.scale_to_fit_width(12.8)
        grp = m.VGroup(label, q)
        if hint:
            h = m.Text(hint, font_size=23, color=m.GREY_A)
            if h.width > 12.8:
                h.scale_to_fit_width(12.8)
            grp.add(h)
        grp.arrange(m.DOWN, buff=0.3).move_to(0.2 * m.UP)
        anims: list[m.Animation] = [m.FadeOut(mob) for mob in old]
        if self.progress not in self.mobjects:
            anims.append(m.FadeIn(self.progress))
        # la question arrive pendant que la diapositive précédente s'efface : pas d'écran noir
        out = m.squish_rate_func(m.smooth, 0.0, 0.35)
        late = m.squish_rate_func(m.smooth, 0.3, 1.0)
        self.play(
            *[m.FadeOut(mob, rate_func=out) for mob in old],
            *[a for a in anims if not isinstance(a, m.FadeOut)],
            *self.set_chapter(chapter),
            m.FadeIn(label, shift=0.2 * m.UP, rate_func=late),
            m.Write(q, rate_func=late),
            *(m.FadeIn(g, rate_func=late) for g in grp[2:]),
            run_time=1.8,
        )
        self.wait(1.3)
        title = title_text(f"{tag} · {question}" if tag[0] in "QE" else question)
        self.play(m.ReplacementTransform(q, title), *[m.FadeOut(g) for g in grp if g is not q], run_time=0.8)
        self.cur_title = title
        return title

    def answer(self, text: str, wait: float = 2.8, color=OK) -> m.VGroup:
        """L'encadré « En une phrase » : la réponse courte de la fiche."""
        box = takeaway_box(text, color)
        box[3].become(m.Text("En une phrase", font_size=16, color=color, weight=m.BOLD).move_to(box[3], aligned_edge=m.LEFT))
        box[2].stretch_to_fit_width(box[3].width + 0.16).move_to(box[3])
        content = [mob for mob in self.mobjects if mob is not self.progress and mob is not self.cur_title]
        anims: list[m.Animation] = [m.FadeIn(box, shift=0.2 * m.UP)]
        if content and self.cur_title is not None:
            g = m.VGroup(*content)
            top = self.cur_title.get_bottom()[1] - 0.15
            bot = box.get_top()[1] + 0.15
            if g.get_bottom()[1] < bot or g.get_top()[1] > top:
                k = min(1.0, (top - bot) / g.height)
                anims.append(g.animate.scale(k).move_to([g.get_center()[0], (top + bot) / 2, 0]))
        self.play(*anims, run_time=0.8)
        self.wait(wait)
        return box

    def bullets(self, lines: list[str], font_size: int = 22, color=m.GREY_A) -> m.VGroup:
        return m.VGroup(*[m.Text(t, font_size=font_size, color=color) for t in lines]).arrange(
            m.DOWN, aligned_edge=m.LEFT, buff=0.2
        )

    # ======================================================================
    # Contexte
    # ======================================================================
    def intro(self) -> None:
        self.next_section("titre")
        t1 = m.Text("De la corrélation spectrale au NFA", font_size=46, weight=m.BOLD)
        t2 = m.Text("Détecter qu'une image a été redimensionnée — réimplémentation 1D de Li et al.", font_size=23, color=ACCENT)
        names = m.Text("Aymen Ouerghi  ·  Mohamed Missaoui", font_size=28)
        proj = m.Text("Projet IMA01 · briques B3 (corrélation) → B4 (a contrario / NFA)", font_size=20, color=DIM)
        fil = m.MathTex(
            r"\rho \;\to\; k(d) \;\to\; \mathcal{B}(n,p)\ \text{sous}\ H_0 \;\to\; \mathrm{NFA}(d) \;\to\; \mathrm{NFA}(d) < \varepsilon",
            font_size=34,
        )
        g = m.VGroup(t1, t2, fil, names, proj).arrange(m.DOWN, buff=0.38)
        self.play(m.Write(t1), run_time=1.3)
        self.play(m.FadeIn(t2, shift=0.2 * m.UP))
        self.play(m.Write(fil), run_time=1.5)
        self.play(m.FadeIn(names), m.FadeIn(proj))
        self.wait(2)

    def context_problem(self) -> None:
        self.ask("Contexte", "Cette photo a-t-elle été agrandie ?", 0, "… sans avoir l'originale sous la main")
        rng = np.random.default_rng(3)
        small = rng.uniform(0.2, 0.95, (6, 6))
        x = np.arange(8) * 6 / 8
        i0 = np.floor(x).astype(int)
        t = x - i0
        i1 = (i0 + 1) % 6
        rows = (1 - t)[:, None] * small[i0] + t[:, None] * small[i1]
        big = (1 - t)[None, :] * rows[:, i0] + t[None, :] * rows[:, i1]

        def grid(vals: np.ndarray, color, side: float = 2.8) -> m.VGroup:
            n = vals.shape[0]
            c = side / n
            return m.VGroup(
                *[
                    m.Square(side_length=c, stroke_width=1, stroke_color=m.GREY_A)
                    .set_fill(color, float(vals[i, j]))
                    .move_to([(j - (n - 1) / 2) * c, ((n - 1) / 2 - i) * c, 0])
                    for i in range(n)
                    for j in range(n)
                ]
            )

        g1 = grid(small, ORIG).shift(3.3 * m.LEFT + 0.6 * m.UP)
        g2 = grid(big, RESAMP).shift(3.3 * m.RIGHT + 0.6 * m.UP)
        l1 = m.Text("image d'origine : M × M", font_size=22).next_to(g1, m.DOWN, buff=0.2)
        l2 = m.Text("image publiée : N × N", font_size=22).next_to(g2, m.DOWN, buff=0.2)
        arrow = m.Arrow(g1.get_right(), g2.get_left(), buff=0.3, color=ACCENT)
        al = m.Text("interpolation", font_size=20).next_to(arrow, m.UP, buff=0.1)
        self.play(m.FadeIn(g1), m.FadeIn(l1))
        self.play(m.GrowArrow(arrow), m.FadeIn(al), m.TransformFromCopy(g1, g2), m.FadeIn(l2), run_time=1.6)
        self.answer("Enjeu en criminalistique : un redimensionnement trahit une retouche ou un montage.\n"
                    "Il faut une trace que l'agrandissement laisse forcément, visible sur la seule image publiée.")

    def context_resampling(self) -> None:
        self.ask("Contexte", "Que fait un redimensionnement ?", 0, "Regardons une seule ligne de pixels : un signal 1D")
        D = self.D
        ch = Chart((0, 20, 5), (-3, 3, 3), 11, 2.5, "x", "").shift(0.4 * m.UP + 0.3 * m.RIGHT)
        su = ch.stems(range(20), D["u"], ORIG, 6)
        sv = ch.stems([j * 0.4 for j in range(50)], D["v"], RESAMP, 2.5)
        e1 = m.MathTex(r"u_0, \dots, u_{M-1}\ \ (M = 100)", font_size=30, color=ORIG)
        e2 = m.MathTex(r"v_j = \sum_k u_k\,\varphi\!\left(j\tfrac{M}{N} - k\right)\ \ (N = 250)", font_size=30, color=RESAMP)
        m.VGroup(e1, e2).arrange(m.RIGHT, buff=0.8).next_to(ch.axes, m.UP, buff=0.3)
        self.play(m.FadeIn(ch), m.LaggedStart(*[m.Create(s) for s in su], lag_ratio=0.04), m.FadeIn(e1))
        self.play(m.LaggedStart(*[m.Create(s) for s in sv], lag_ratio=0.02), m.FadeIn(e2), run_time=1.8)
        self.answer("On interpole les M valeurs par un noyau φ, puis on relit N valeurs : v est fabriqué à partir de u.\n"
                    "De l'information est dupliquée — reste à savoir où cela se voit.")

    # ======================================================================
    # Partie A — B3
    # ======================================================================
    def q1_fourier(self) -> None:
        self.ask("Q1", "Pourquoi chercher la trace dans Fourier ?", 1, "plutôt que dans le domaine spatial, comme les méthodes classiques")
        D = self.D
        a = Chart((0, 100, 50), (0, 0.3, 0.3), 3.3, 1.9, "n", r"|\tilde u_n|").shift(4.45 * m.LEFT + 1.0 * m.UP)
        b = Chart((0, 250, 50), (0, 0.3, 0.3), 7.4, 1.9, "n", r"|\tilde v_n|").shift(2.6 * m.RIGHT + 1.0 * m.UP)
        lab_a = m.Text("spectre de u (M = 100)", font_size=20, color=ORIG).next_to(a.axes, m.UP, buff=0.1)
        lab_b = m.Text("spectre de v (N = 250)", font_size=20, color=RESAMP).next_to(b.axes, m.UP, buff=0.1)
        sa = a.stems(range(100), D["absU"], ORIG, 2)
        sb = b.stems(range(250), D["absV"], RESAMP, 1.6)
        self.play(m.FadeIn(a), m.FadeIn(lab_a), m.Create(sa), run_time=1.2)
        self.play(m.FadeIn(b), m.FadeIn(lab_b), m.Create(sb), run_time=1.5)
        boxes = m.VGroup(*[m.SurroundingRectangle(m.VGroup(*sb[s : s + 10]), color=OK, buff=0.06, stroke_width=3) for s in (40, 140)])
        src = m.SurroundingRectangle(m.VGroup(*sa[40:50]), color=OK, buff=0.06, stroke_width=3)
        self.play(m.Create(src))
        self.play(m.TransformFromCopy(src, boxes[0]), m.TransformFromCopy(src, boxes[1]), run_time=1.2)
        y = b.p(0, 0)[1] - 0.8
        arrow = m.DoubleArrow([boxes[0].get_center()[0], y, 0], [boxes[1].get_center()[0], y, 0], buff=0, color=OK, stroke_width=3)
        d_lab = m.MathTex(r"d = 100 = M \bmod N", font_size=30, color=OK).next_to(arrow, m.DOWN, buff=0.1)
        self.play(m.GrowFromCenter(arrow), m.Write(d_lab))
        thm = m.VGroup(
            m.Text("Théorème 1 (repliement spectral) :", font_size=20, color=m.GREY_A),
            m.MathTex(r"\tilde v_n = M \sum_{z} c_{zN+n}(\varphi)\, \tilde u_{zN+n}", font_size=32),
        ).arrange(m.RIGHT, buff=0.3).next_to(d_lab, m.DOWN, buff=0.25)
        self.play(m.FadeIn(thm))
        self.answer("Le rééchantillonnage réplique et module le spectre : la trace y apparaît comme une corrélation\n"
                    "entre zones du spectre, séparées de d = ±M mod N.")

    def q2_q3_patches(self) -> None:
        self.ask("Q2 · Q3", "On découpe l'image ou le spectre ? Et pourquoi en patches ?", 1)
        D = self.D
        steps = m.VGroup(
            m.Text("signal / image", font_size=22, color=ORIG),
            m.MathTex(r"\xrightarrow{\ \mathrm{FFT}\ }", font_size=36),
            m.Text("spectre", font_size=22, color=RESAMP),
            m.MathTex(r"\xrightarrow{\ \text{découpe}\ }", font_size=36),
            m.Text("25 patches de L = 10 fréquences", font_size=22, color=OK),
        ).arrange(m.RIGHT, buff=0.25).shift(2.3 * m.UP)
        self.play(m.LaggedStart(*[m.FadeIn(s) for s in steps], lag_ratio=0.3), run_time=1.6)
        ch = Chart((0, 250, 50), (0, 0.3, 0.3), 11, 1.7, "n", r"|\tilde v_n|").shift(0.6 * m.UP + 0.3 * m.RIGHT)
        sb = ch.stems(range(250), D["absV"], RESAMP, 1.6)
        cuts = m.VGroup(*[m.DashedLine(ch.p(10 * i, 0), ch.p(10 * i, 0.3), color=OK, stroke_width=1.5) for i in range(1, 25)])
        self.play(m.FadeIn(ch), m.Create(sb), run_time=1.2)
        self.play(m.LaggedStart(*[m.Create(c) for c in cuts], lag_ratio=0.05), run_time=1.2)
        why = m.VGroup(
            m.Text("Pourquoi pas point par point ?  Un point = 1 seul échantillon par fréquence : trop bruité.", font_size=21),
            m.Text("Cas extrême mesuré (E1) : avec L = 2, |PCC| vaut toujours 1 → aucune information.", font_size=21, color=BAD),
            m.Text("Un patch moyenne L valeurs : la mesure se stabilise et la vraie corrélation ressort.", font_size=21, color=OK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18).next_to(ch, m.DOWN, buff=0.3)
        self.play(m.LaggedStart(*[m.FadeIn(w, shift=0.15 * m.RIGHT) for w in why], lag_ratio=0.5), run_time=2.2)
        self.answer("On découpe le SPECTRE, pas l'image : l'image ne sert qu'à calculer la FFT.\n"
                    "Un point du spectre est une mesure trop bruitée ; un patch est une moyenne robuste.")

    def q4_pcc(self) -> None:
        self.ask("Q4", "Comment mesure-t-on la ressemblance entre deux patches ?", 1)
        f = m.MathTex(
            r"\rho(x, y) = \frac{\big|\langle x - \bar x,\ y - \bar y \rangle\big|}{\|x - \bar x\|_2\ \|y - \bar y\|_2}\ \in [0, 1]\qquad(\text{éq. } 16)",
            font_size=38,
        ).shift(1.9 * m.UP)
        self.play(m.Write(f), run_time=1.4)
        recipe = m.VGroup(
            *[
                m.VGroup(m.Text(n, font_size=22, color=ACCENT, weight=m.BOLD), m.Text(t, font_size=21)).arrange(m.RIGHT, buff=0.2)
                for n, t in (
                    ("1.", "aplatir chaque patch en vecteur"),
                    ("2.", "centrer : retirer la moyenne"),
                    ("3.", "produit hermitien ⟨x, y⟩ = Σ xᵢ · conj(yᵢ)"),
                    ("4.", "normaliser par les normes, prendre le module"),
                )
            ]
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).next_to(f, m.DOWN, buff=0.35).shift(2.6 * m.LEFT)
        cos = m.Text("= le cosinus de l'angle\nentre deux vecteurs", font_size=21, color=m.GREY_A).next_to(recipe, m.RIGHT, buff=1.0)
        self.play(m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in recipe], lag_ratio=0.35), run_time=2)
        self.play(m.FadeIn(cos))
        D = self.D
        ex = m.VGroup(
            m.MathTex(rf"\rho(P_{{40:50}},\ P_{{140:150}}) = {D['pcc_copy']:.2f}", font_size=30, color=OK),
            m.MathTex(rf"\rho(P_{{40:50}},\ P_{{77:87}}) = {D['pcc_other']:.2f}", font_size=30, color=DIM),
        ).arrange(m.RIGHT, buff=0.9).next_to(recipe, m.DOWN, buff=0.35).set_x(0)
        tags = m.VGroup(
            m.Text("copie (d = 100) — notre code", font_size=17, color=OK).next_to(ex[0], m.DOWN, buff=0.08),
            m.Text("sans rapport (d = 37)", font_size=17, color=DIM).next_to(ex[1], m.DOWN, buff=0.08),
        )
        self.play(m.FadeIn(ex), m.FadeIn(tags))
        self.answer("Par le coefficient de corrélation de Pearson complexe (PCC) :\n1 = même forme, 0 = indépendants.")

    def q5_q6_conj_abs(self) -> None:
        self.ask("Q5 · Q6", "Pourquoi conj et abs ? Le PCC perd-il de l'information ?", 1)
        left = m.VGroup(
            m.Text("Sans conjugué", font_size=24, weight=m.BOLD, color=BAD),
            m.MathTex(r"\langle i, i\rangle = i \cdot i = -1", font_size=34),
            m.Text("« se corréler avec soi-même »\nne donnerait même pas 1", font_size=19, color=m.GREY_A),
            m.Text("Avec conjugué", font_size=24, weight=m.BOLD, color=OK),
            m.MathTex(r"\langle i, i\rangle = i \cdot \bar i = 1", font_size=34),
        ).arrange(m.DOWN, buff=0.2).shift(3.4 * m.LEFT + 0.6 * m.UP)
        right = m.VGroup(
            m.Text("Le module rend ρ invariant", font_size=24, weight=m.BOLD, color=ACCENT),
            m.MathTex(r"Q \approx c \cdot P,\quad c \in \mathbb{C}\ \ (\text{modulation du filtre})", font_size=30),
            m.MathTex(r"\Rightarrow\ \rho(P, Q) = 1\ \ \text{quel que soit } c", font_size=30, color=OK),
            m.Text("on jette volontairement moyenne, amplitude et phase :\ncette perte EST l'invariance recherchée", font_size=19, color=m.GREY_A),
            m.Text("testé : test_invariance_echelle_et_phase (c = 3, 2i, −1+i)", font_size=17, color=DIM, font="monospace"),
        ).arrange(m.DOWN, buff=0.22).shift(3.0 * m.RIGHT + 0.6 * m.UP)
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in left], lag_ratio=0.3), run_time=2.2)
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in right], lag_ratio=0.3), run_time=2.4)
        lim = m.Text("Limites assumées : pas invariant à la rotation ; M n'est connu qu'à ±M mod N près.", font_size=19, color=BAD)
        lim.next_to(m.VGroup(left, right), m.DOWN, buff=0.3)
        self.play(m.FadeIn(lim))
        self.answer("conj → le produit devient une vraie norme (l'autocorrélation vaut 1) ;\n"
                    "abs → un score réel ET invariant à la phase.  Le PCC reconnaît la copie quel que soit le noyau.")

    def q7_why_pcc(self) -> None:
        self.ask("Q7", "Pourquoi le PCC plutôt qu'une autre mesure ?", 1)
        rows = [
            ("MSE", "✗", "—", "sensible à la modulation c"),
            ("produit scalaire", "✗", "—", "dépend de l'échelle"),
            ("cosine", "≈", "—", "presque le PCC, sans centrage"),
            ("information mutuelle", "—", "✗", "coûteuse, sans loi simple"),
            ("PCC", "✓", "✓", "simple, invariant, binomiale exacte"),
        ]
        head = ["mesure", "invariante à c·P", "loi sous H₀ connue", "remarque"]
        table = m.VGroup()
        widths = [3.0, 2.4, 2.8, 4.4]
        for r_i, row in enumerate([head] + rows):
            cells = m.VGroup()
            for c_i, (txt, w) in enumerate(zip(row, widths)):
                color = DIM if r_i == 0 else (OK if txt == "✓" else BAD if txt == "✗" else ACCENT if txt == "≈" else m.WHITE)
                if r_i == len(rows) and c_i == 0:
                    color = OK
                t = m.Text(txt, font_size=20, color=color, weight=m.BOLD if (r_i == 0 or r_i == len(rows)) else m.NORMAL)
                cell = m.Rectangle(width=w, height=0.5, stroke_width=0)
                t.move_to(cell)
                cells.add(m.VGroup(cell, t))
            cells.arrange(m.RIGHT, buff=0)
            table.add(cells)
        table.arrange(m.DOWN, buff=0.05).shift(0.6 * m.UP)
        sep = m.Line(table[0].get_corner(m.DL), table[0].get_corner(m.DR), color=DIM, stroke_width=1.5)
        hl = m.SurroundingRectangle(table[-1], color=OK, buff=0.04)
        self.play(m.FadeIn(table[0]), m.Create(sep))
        self.play(m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in table[1:]], lag_ratio=0.3), run_time=2.2)
        self.play(m.Create(hl))
        self.answer("C'est la mesure la plus simple qui soit invariante aux bonnes nuisances ET dont on connaît la loi :\n"
                    "avec le test du maximum local, elle donne une binomiale exacte sous H₀ (indispensable en B4).")

    def q10_key_result(self) -> None:
        self.ask("Q10", "Quel résultat clé avons-nous validé en 1D ?", 1, "bruit blanc M = 100 → N = 250, plus proche voisin, L = 10")
        D = self.D
        ch = Chart((0, 250, 50), (0, 1, 0.5), 10.8, 3.0, "d", r"\bar\rho(d)").shift(0.55 * m.UP + 0.3 * m.RIGHT)
        d = list(range(1, 250))
        lg = m.VGroup(
            m.Text("signal rééchantillonné", font_size=19, color=RESAMP),
            m.Text("bruit blanc non rééchantillonné", font_size=19, color=DIM),
        ).arrange(m.RIGHT, buff=0.5).next_to(ch.axes, m.UP, buff=0.15)
        self.play(m.FadeIn(ch), m.FadeIn(lg))
        self.play(m.Create(ch.line(d, D["rho_ref_mean"][1:], DIM, 2)), run_time=1)
        self.play(m.Create(ch.line(d, D["rho_mean"][1:], RESAMP, 2.5)), run_time=1.6)
        pk, rf = D["rho_mean"][100], D["rho_ref_mean"][100]
        dot1 = m.Dot(ch.p(100, pk), color=OK, radius=0.08)
        dot2 = m.Dot(ch.p(100, rf), color=DIM, radius=0.07)
        l1 = m.MathTex(rf"\bar\rho(100) = {pk:.2f}", font_size=28, color=OK).next_to(dot1, m.UR, buff=0.08)
        l2 = m.MathTex(rf"{rf:.2f}", font_size=26, color=DIM).next_to(dot2, m.DR, buff=0.08)
        self.play(m.FadeIn(dot1, scale=1.5), m.FadeIn(dot2, scale=1.5), m.Write(l1), m.Write(l2))
        harm = m.Text("autres pics : 50, 150, 200 = ±kM mod N (harmoniques)", font_size=20, color=m.GREY_A).next_to(ch, m.DOWN, buff=0.15)
        self.play(m.FadeIn(harm))
        self.answer(f"Pic net à d = M mod N = 100 : ρ ≈ {pk:.2f} contre ≈ {rf:.2f} sans rééchantillonnage,\n"
                    "exactement où la théorie le prédit.  C'est le cœur du papier, reproduit from scratch en 1D.")

    # ======================================================================
    # Partie B — B4
    # ======================================================================
    def q11_no_threshold(self) -> None:
        self.ask("Q11", "Pourquoi ne pas simplement seuiller ρ > 0,5 ?", 2)
        D = self.D
        d = list(range(1, 250))
        a = Chart((0, 250, 100), (0, 1, 0.5), 5.2, 2.6, "d", r"\bar\rho").shift(3.4 * m.LEFT + 0.5 * m.UP)
        b = Chart((0, 250, 100), (0, 1, 0.5), 5.2, 2.6, "d", r"\bar\rho").shift(3.6 * m.RIGHT + 0.5 * m.UP)
        ta = m.Text("L = 10 (bruit blanc pur)", font_size=20).next_to(a.axes, m.UP, buff=0.15)
        tb = m.Text("L = 4 (bruit blanc pur)", font_size=20).next_to(b.axes, m.UP, buff=0.15)
        self.play(m.FadeIn(a), m.FadeIn(b), m.FadeIn(ta), m.FadeIn(tb))
        self.play(m.Create(a.line(d, D["rho_ref_mean"][1:], DIM, 2)), m.Create(b.line(d, D["rho4_ref_mean"][1:], DIM, 2)), run_time=1.5)
        self.play(m.Create(a.hline(0.5)), m.Create(b.hline(0.5)))
        fa = float(np.median(D["rho_ref_mean"][1:]))
        fb = float(np.median(D["rho4_ref_mean"][1:]))
        ra = m.MathTex(rf"\text{{fond}} \approx {fa:.2f}", font_size=28, color=OK).next_to(a, m.DOWN, buff=0.15)
        rb = m.MathTex(rf"\text{{fond}} \approx {fb:.2f} > 0{{,}}5\ \Rightarrow\ \text{{faux positifs !}}", font_size=28, color=BAD).next_to(b, m.DOWN, buff=0.15)
        self.play(m.Write(ra), m.Write(rb))
        self.answer("Parce que le bruit de fond n'est pas fixe : il dépend de L, de la taille, du contenu…\n"
                    "Un seuil universel sur ρ crée des faux positifs ou rate des traces selon le régime.")

    def q12_background(self) -> None:
        self.ask("Q12", "Pourquoi le bruit de fond vaut ~0,25 et pas 0 ?", 2, "Deux vecteurs indépendants ont-ils une corrélation empirique nulle ?")
        E = self.R["E1"]
        L = E["L"]
        ch = Chart((0, 52, 10), (0, 1, 0.5), 7.2, 3.2, "L", r"\bar\rho_{\text{fond}}").shift(2.2 * m.LEFT + 0.45 * m.UP)
        xs = np.linspace(2, 50, 60)
        meas = ch.line(L, E["fond"], ORIG)
        theo = ch.line(xs, [1 / math.sqrt(x) for x in xs], ACCENT, 2)
        lg = m.VGroup(m.Text("mesuré (E1)", font_size=19, color=ORIG), m.MathTex(r"1/\sqrt{L}", font_size=28, color=ACCENT)).arrange(m.RIGHT, buff=0.4)
        lg.next_to(ch.axes, m.UP, buff=0.15)
        self.play(m.FadeIn(ch), m.FadeIn(lg))
        self.play(m.Create(meas), m.Create(theo), run_time=1.6)
        side = m.VGroup(
            m.Text("Nulle en espérance…", font_size=22),
            m.Text("… mais elle fluctue en ~1/√L\nsur un échantillon fini.", font_size=22, color=ACCENT),
            m.MathTex(r"L = 10:\ 1/\sqrt{10} \approx 0{,}32", font_size=28),
            m.MathTex(r"L = 4:\ 1/\sqrt{4} = 0{,}5", font_size=28),
            m.Text("Une loi statistique,\npas un artefact machine.", font_size=20, color=OK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(ch, m.RIGHT, buff=0.6)
        self.play(m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.LEFT) for s in side], lag_ratio=0.35), run_time=2.4)
        self.answer("Deux vecteurs indépendants de longueur finie ont une corrélation empirique ~1/√L, pas 0.")

    def q13_count_votes(self) -> None:
        self.ask("Q13", "Faut-il moyenner ρ, ou compter ?", 2, "Deux scénarios de corrélations sur 10 patches (schéma)")
        A = [0.95, 0.2, 0.25, 0.15, 0.3, 0.2, 0.25, 0.2, 0.3, 0.2]
        B = [0.32, 0.31, 0.34, 0.30, 0.33, 0.32, 0.31, 0.33, 0.30, 0.34]

        def panel(vals, name, color, shift):
            ch = Chart((0, 10, 5), (0, 1, 0.5), 4.6, 2.3, "i", r"\rho(P_i, d)", x_numbers=False).shift(shift)
            bars = m.VGroup(
                *[
                    m.Rectangle(width=0.32, height=max(0.01, ch.p(0, v)[1] - ch.p(0, 0)[1]), stroke_width=0)
                    .set_fill(color, 0.85)
                    .move_to((ch.p(i + 0.5, 0) + ch.p(i + 0.5, v)) / 2)
                    for i, v in enumerate(vals)
                ]
            )
            t = m.Text(name, font_size=21, color=color).next_to(ch.axes, m.UP, buff=0.15)
            mean = m.MathTex(rf"\text{{moyenne}} = {np.mean(vals):.2f}", font_size=28).next_to(ch, m.DOWN, buff=0.15)
            return ch, bars, t, mean

        ca, ba, tA, mA = panel(A, "A : un seul patch très corrélé", BAD, 3.4 * m.LEFT + 0.75 * m.UP)
        cb, bb, tB, mB = panel(B, "B : tous d'accord à la même distance", OK, 3.6 * m.RIGHT + 0.75 * m.UP)
        self.play(m.FadeIn(ca), m.FadeIn(cb), m.FadeIn(tA), m.FadeIn(tB))
        self.play(m.LaggedStart(*[m.GrowFromEdge(x, m.DOWN) for x in ba], lag_ratio=0.05), m.LaggedStart(*[m.GrowFromEdge(x, m.DOWN) for x in bb], lag_ratio=0.05), run_time=1.6)
        self.play(m.Write(mA), m.Write(mB))
        v = m.VGroup(
            m.Text("même moyenne : la moyenne ne les distingue pas", font_size=21, color=m.GREY_A),
            m.Text("A arrive par hasard ; B est quasi impossible par hasard → preuve bien plus forte", font_size=21, color=OK),
        ).arrange(m.DOWN, buff=0.15).next_to(m.VGroup(mA, mB), m.DOWN, buff=0.3)
        self.play(m.FadeIn(v))
        self.answer("On compte le NOMBRE de patches qui « votent » pour une distance d ; on ne moyenne pas ρ.")

    def q14_helmholtz(self) -> None:
        self.ask("Q14", "Quel est le principe fondateur ?", 2, "le principe de Helmholtz — cadre a contrario")
        rng = np.random.default_rng(5)
        pts = rng.uniform(-2.2, 2.2, (60, 2))
        left = m.VGroup(*[m.Dot([x - 3.4, y + 0.55, 0], radius=0.05, color=DIM) for x, y in pts])
        al = [m.Dot([-2.0 + 0.4 * i + 3.4, -1.2 + 0.32 * i + 0.55, 0], radius=0.06, color=OK) for i in range(10)]
        right = m.VGroup(*[m.Dot([x + 3.4, y + 0.55, 0], radius=0.05, color=DIM) for x, y in pts[:50]], *al)
        tl = m.Text("rien de remarquable", font_size=20, color=DIM).next_to(left, m.DOWN, buff=0.2)
        tr = m.Text("10 points alignés : trop improbable par hasard", font_size=20, color=OK).next_to(right, m.DOWN, buff=0.2)
        self.play(m.FadeIn(left), m.FadeIn(right), run_time=1.2)
        self.play(m.FadeIn(tl), m.FadeIn(tr), m.Indicate(m.VGroup(*al), color=OK))
        h0 = m.MathTex(r"H_0:\ \text{« l'image n'a pas été rééchantillonnée »}", font_size=32).next_to(m.VGroup(tl, tr), m.DOWN, buff=0.35)
        self.play(m.Write(h0))
        self.answer("Une structure est perçue quand elle s'écarte trop du hasard : on cherche ce qui serait\n"
                    "trop improbable sous H₀.  Plus c'est improbable par hasard, plus la preuve est forte.")

    def q15_q16_vote(self) -> None:
        self.ask("Q15 · Q16", "Qu'est-ce que r, et quelle est la probabilité d'un vote par hasard ?", 2)
        D = self.D
        win = list(range(80, 121))
        one = Chart((80, 120, 10), (0, 1, 0.5), 6.2, 2.4, "d'", r"\rho(P_4, d')").shift(2.6 * m.LEFT + 0.75 * m.UP)
        cur = one.line(win, [D["rho_patch4"][x] for x in win], ORIG, 2.5)
        top = m.Dot(one.p(100, D["rho_patch4"][100]), color=OK, radius=0.09)
        br = m.BraceBetweenPoints(one.p(80, 1.05), one.p(120, 1.05), direction=m.UP, color=ACCENT)
        brl = m.MathTex(r"2r + 1 = 41\ \text{distances candidates}", font_size=26, color=ACCENT).next_to(br, m.UP, buff=0.08)
        vote = m.Text("max en d = 100 → le patch 4 « vote » pour 100", font_size=20, color=OK).next_to(one, m.DOWN, buff=0.15)
        self.play(m.FadeIn(one), m.Create(cur), run_time=1.2)
        self.play(m.GrowFromCenter(br), m.FadeIn(brl))
        self.play(m.FadeIn(top, scale=1.6), m.FadeIn(vote))
        side = m.VGroup(
            m.Text("Sous H₀, aucune des 41\ncandidates n'est privilégiée :", font_size=21),
            m.MathTex(r"p = \frac{1}{2r+1}\qquad(\text{éq. } 20)", font_size=36, color=ACCENT),
            m.MathTex(r"r = 20\ \Rightarrow\ p = \tfrac{1}{41} \approx 0{,}024", font_size=30),
            m.Text("≈ 1 patch sur 41 vote\npour d par pur hasard", font_size=20, color=m.GREY_A),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.25).next_to(one, m.RIGHT, buff=0.7)
        self.play(m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.LEFT) for s in side], lag_ratio=0.35), run_time=2.2)
        self.answer("r définit la fenêtre [d − r, d + r] (r = 20 → 41 candidates) ; un patch vote pour d si sa\n"
                    "corrélation y est maximale.  Par pur hasard : p = 1/(2r+1).")

    def q17_binomial(self) -> None:
        self.ask("Q17", "Quelle loi suit le nombre de votes k(d) ?", 2)
        D = self.D
        n, r = D["n_patches"], 20
        kk = D["k"]
        votes = Chart((0, 250, 50), (0, 25, 5), 10.4, 2.6, "d", "k(d)").shift(1.0 * m.UP + 0.3 * m.RIGHT)
        bars = votes.stems(range(r, 250 - r), kk[r : 250 - r], RESAMP, 3)
        exp_line = votes.hline(n / (2 * r + 1), color=ACCENT)
        vl = m.Text(f"votes observés sur nos {n} patches (signal rééchantillonné)", font_size=19, color=RESAMP).next_to(votes.axes, m.UP, buff=0.12)
        self.play(m.FadeIn(votes), m.FadeIn(vl), m.LaggedStart(*[m.Create(b) for b in bars], lag_ratio=0.005), run_time=1.6)
        el = m.MathTex(rf"np \approx {n / 41:.1f}", font_size=26, color=ACCENT).next_to(votes.p(250, n / 41), m.RIGHT, buff=0.1)
        self.play(m.Create(exp_line), m.FadeIn(el))
        law = m.VGroup(
            m.MathTex(r"k(d) \sim \mathcal{B}\!\left(n,\ \tfrac{1}{2r+1}\right)\ \ \text{sous}\ H_0", font_size=32),
            m.MathTex(
                rf"k(100) = {int(kk[100])}\ \text{{votes}}\ \Rightarrow\ \mathbb{{P}}_{{H_0}}\big[\mathbf{{k}} \ge {int(kk[100])}\big] \approx 10^{{{D['p_k100']:.0f}}}",
                font_size=32,
                color=OK,
            ),
        ).arrange(m.RIGHT, buff=0.8).next_to(votes, m.DOWN, buff=0.3)
        if law.width > 13:
            law.arrange(m.DOWN, buff=0.15).next_to(votes, m.DOWN, buff=0.25)
        self.play(m.Write(law[0]))
        self.play(m.Write(law[1]))
        why = m.Text("L'indépendance vient de patches NON chevauchants — d'où ce choix.", font_size=20, color=m.GREY_A).next_to(law, m.DOWN, buff=0.2)
        self.play(m.FadeIn(why))
        self.answer("k(d) suit une loi binomiale B(n, p) : n = nombre de patches, p = 1/(2r+1).")

    def q18_multiple_tests(self) -> None:
        self.ask("Q18", "Quel est le problème des tests multiples ?", 2, "On teste toutes les distances d = 1 … ~N/2 en même temps")
        grid = m.VGroup(*[m.Square(side_length=0.22, stroke_width=1, stroke_color=m.GREY_D) for _ in range(300)])
        grid.arrange_in_grid(rows=10, cols=30, buff=0.05).shift(1.0 * m.UP)
        rng = np.random.default_rng(2)
        bad = rng.choice(300, 3, replace=False)
        lab = m.Text("300 cas sains, testés par un test qui se trompe 1 fois sur 100", font_size=21).next_to(grid, m.UP, buff=0.2)
        self.play(m.FadeIn(lab), m.LaggedStart(*[m.FadeIn(s) for s in grid], lag_ratio=0.002), run_time=1.4)
        self.play(*[grid[i].animate.set_fill(BAD, 1) for i in bad], run_time=0.8)
        res = m.MathTex(r"300 \times \tfrac{1}{100} = 3\ \text{fausses alarmes en moyenne}", font_size=34, color=BAD).next_to(grid, m.DOWN, buff=0.3)
        self.play(m.Write(res))
        same = m.Text("Même sur une image authentique, quelques distances crieront « rééchantillonné » par hasard.", font_size=20, color=m.GREY_A)
        same.next_to(res, m.DOWN, buff=0.2)
        self.play(m.FadeIn(same))
        self.answer("Tester ~300 distances multiplie les occasions de faux positif :\nle danger n'est pas une distance isolée, c'est la multiplication des tests.", color=ACCENT)

    def q19_nfa(self) -> None:
        self.ask("Q19", "Qu'est-ce que le NFA, et comment le calcule-t-on ?", 2)
        f = m.MathTex(
            r"\mathrm{NFA}(d) = N_{\text{distances}} \times \mathbb{P}_{H_0}\big[\mathbf{k} \ge k_{\text{obs}}(d)\big],\quad \mathbf{k} \sim \mathcal{B}\big(n, \tfrac{1}{2r+1}\big)\qquad(\text{éq. } 21)",
            font_size=32,
        ).shift(2.2 * m.UP)
        self.play(m.Write(f), run_time=1.5)
        ex = m.VGroup(
            m.MathTex(r"P = 1/100:\quad \mathrm{NFA} = 300 \times 10^{-2} = 3\ \Rightarrow\ \text{rejetée}", font_size=30, color=BAD),
            m.MathTex(r"P = 10^{-6}:\quad \mathrm{NFA} = 300 \times 10^{-6} = 3\cdot 10^{-4}\ \Rightarrow\ \text{validée}", font_size=30, color=OK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).next_to(f, m.DOWN, buff=0.35)
        self.play(m.LaggedStart(*[m.FadeIn(e, shift=0.2 * m.RIGHT) for e in ex], lag_ratio=0.5), run_time=1.6)
        src = inspect.getsource(compute_nfa)
        src = "\n".join(line for line in src.splitlines() if '"""' not in line)
        code = m.Code(code_string=src, language="python", add_line_numbers=False)
        code.scale_to_fit_height(2.3)
        if code.width > 8.5:
            code.scale_to_fit_width(8.5)
        code.next_to(ex, m.DOWN, buff=0.3)
        cap = m.Text("notre src/acontrario.py — la même formule", font_size=15, color=DIM, font="monospace").next_to(code, m.DOWN, buff=0.08)
        self.play(m.FadeIn(code), m.FadeIn(cap))
        self.answer("NFA(d) = (nombre de distances testées) × P(k ≥ k_observé) = nombre attendu de fausses alarmes.\n"
                    "Le vrai signal passe, le bruit multiplié ne passe plus.")

    def q20_decision(self) -> None:
        self.ask("Q20", "Que signifie exactement NFA(d) < ε ?", 2, "Notre courbe NFA(d) — l'équivalent 1D de la Fig. 2 du papier")
        D = self.D
        nf = Chart((0, 250, 50), (-30, 5, 10), 10.6, 3.4, "d", r"\log_{10}\mathrm{NFA}(d)").shift(0.45 * m.UP + 0.3 * m.RIGHT)
        xs = [i for i, y in enumerate(D["lognfa"]) if y is not None]
        thr = nf.hline(0)
        tl = m.MathTex(r"\varepsilon = 1", font_size=26, color=BAD).next_to(nf.p(250, 0), m.RIGHT, buff=0.1)
        legend = m.VGroup(
            m.Text("signal rééchantillonné ×2,5", font_size=20, color=RESAMP), m.Text("référence (non rééchantillonné)", font_size=20, color=DIM)
        ).arrange(m.RIGHT, buff=0.5).next_to(nf.axes, m.UP, buff=0.15)
        self.play(m.FadeIn(nf), m.Create(thr), m.FadeIn(tl), m.FadeIn(legend))
        self.play(m.Create(nf.line(xs, [D["lognfa_ref"][i] for i in xs], DIM, 2, clip=(-30, 5))), run_time=1.2)
        self.play(m.Create(nf.line(xs, [D["lognfa"][i] for i in xs], RESAMP, 2.5, clip=(-30, 5))), run_time=1.8)
        strong = [k for k in D["det"] if D["lognfa"][k] < -5]
        marks = m.VGroup()
        for k in strong:
            dot = m.Dot(nf.p(k, max(D["lognfa"][k], -30)), color=OK, radius=0.07)
            marks.add(m.VGroup(dot, m.MathTex(str(k), font_size=24, color=OK).next_to(dot, m.LEFT, buff=0.1)))
        self.play(m.LaggedStart(*[m.FadeIn(mk, scale=1.4) for mk in marks], lag_ratio=0.2))
        note = m.Text(
            "Ce n'est PAS « zéro faux positif » : c'est un budget moyen de fausses alarmes par signal.",
            font_size=20,
            color=m.GREY_A,
        ).next_to(nf, m.DOWN, buff=0.15)
        self.play(m.FadeIn(note))
        self.answer("On valide d si NFA(d) < ε : le nombre moyen de fausses alarmes est borné par ε (typiquement 1).\n"
                    "Ici les 4 harmoniques passent avec NFA ≈ 10⁻²⁰ à 10⁻²⁵.")

    # ======================================================================
    # Notre code et nos résultats
    # ======================================================================
    def code_and_tests(self) -> None:
        self.ask("Code", "Comment l'avons-nous programmé et vérifié ?", 3, "Un module par brique, chacun testé")
        steps = [
            ("resampling.py", "resample_1d", "rééchantillonner"),
            ("dft.py", "dft", "spectre (éq. 1)"),
            ("spectral.py", "patch_correlations_1d", "ρ(Pᵢ, d) — B3"),
            ("acontrario.py", "count_votes · compute_nfa", "k(d), NFA — B4"),
            ("acontrario.py", "decide", "NFA < ε"),
        ]
        boxes = m.VGroup()
        for fname, fn, what in steps:
            content = m.VGroup(
                m.Text(fname, font_size=16, color=DIM, font="monospace"),
                m.Text(fn, font_size=15, color=ACCENT, font="monospace"),
                m.Text(what, font_size=18),
            ).arrange(m.DOWN, buff=0.12)
            boxes.add(m.VGroup(m.RoundedRectangle(width=max(2.2, content.width + 0.3), height=1.35, corner_radius=0.12, color=ORIG), content))
        boxes.arrange(m.RIGHT, buff=0.32)
        if boxes.width > 13.2:
            boxes.scale_to_fit_width(13.2)
        boxes.shift(1.6 * m.UP)
        arrows = m.VGroup(*[m.Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.03, stroke_width=3) for i in range(4)])
        self.play(m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.RIGHT) for b in boxes], lag_ratio=0.25), m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.25), run_time=2.2)
        tests = m.VGroup(
            m.Text("10 tests pytest, tous verts", font_size=24, color=OK, weight=m.BOLD),
            *[
                m.Text("✓ " + t, font_size=19)
                for t in (
                    "DFT puis IDFT = identité ; composante continue = moyenne",
                    "×2 plus proche voisin = formule spectrale théorique",
                    "PCC(x, x) = 1 ; PCC(x, c·x) = 1 ; vecteurs indépendants < 0,2",
                    "pic de ρ̄ à d = M mod N, et pas de pic sans rééchantillonnage",
                    "vrai positif (M mod N détecté) ; vrai négatif (rien, ε = 10⁻³)",
                )
            ],
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16).next_to(boxes, m.DOWN, buff=0.45)
        self.play(m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.RIGHT) for t in tests], lag_ratio=0.2), run_time=1.8)
        self.wait(3)

    def e1_patch_size(self) -> None:
        self.ask("E1", "Quelle taille de patch L choisir ?", 4, "M = 100, N = 250, r = 20, plus proche voisin")
        E = self.R["E1"]
        L = E["L"]
        a = Chart((0, 52, 10), (0, 1, 0.5), 5.0, 2.9, "L", r"\bar\rho").shift(3.3 * m.LEFT + 0.55 * m.UP)
        b = Chart((0, 52, 10), (-35, 5, 10), 5.0, 2.9, "L", r"\log_{10}\mathrm{NFA}").shift(3.7 * m.RIGHT + 0.55 * m.UP)
        la = m.VGroup(m.Text("pic (d = 100)", font_size=18, color=OK), m.Text("bruit de fond", font_size=18, color=ORIG)).arrange(m.RIGHT, buff=0.3).next_to(a.axes, m.UP, buff=0.2)
        lb = m.VGroup(m.Text("NFA au pic", font_size=18, color=RESAMP), m.Text("meilleur NFA référence", font_size=18, color=DIM)).arrange(m.RIGHT, buff=0.3).next_to(b.axes, m.UP, buff=0.2)
        self.play(m.FadeIn(a), m.FadeIn(b), m.FadeIn(la), m.FadeIn(lb))
        self.play(m.Create(a.line(L, E["fond"], ORIG)), m.Create(a.line(L, E["pic"], OK)), run_time=1.4)
        self.play(m.Create(b.hline(0)), m.Create(b.line(L, E["nfa_ref"], DIM)), m.Create(b.line(L, E["nfa_pic"], RESAMP)), run_time=1.4)
        i = int(np.argmin(E["nfa_pic"]))
        best = m.Dot(b.p(L[i], E["nfa_pic"][i]), color=ACCENT, radius=0.09)
        self.play(m.FadeIn(best, scale=1.5), m.Write(m.MathTex(rf"L={L[i]}:\ {E['nfa_pic'][i]:.1f}", font_size=26, color=ACCENT).next_to(best, m.RIGHT, buff=0.15)))
        self.answer("Compromis (Q3 + Q12) : L petit → beaucoup de votants mais un fond élevé ;\nL grand → fond bas mais peu de votants.  Optimum vers L ≈ 4 à 8.")

    def e2_radius(self) -> None:
        self.ask("E2", "Quel rayon de voisinage r ?", 4, "L = 8 — r change p = 1/(2r+1), donc toute la binomiale")
        E = self.R["E2"]
        rs = E["r"]
        ch = Chart((0, 85, 20), (-40, 5, 10), 7.0, 3.5, "r", r"\log_{10}\mathrm{NFA}").shift(2.2 * m.LEFT + 0.45 * m.UP)
        dots = ch.dots(rs, E["nfa_pic"], [RESAMP] * len(rs), 0.07)
        lg = m.VGroup(m.Text("NFA au pic d = 100", font_size=18, color=RESAMP), m.Text("meilleur NFA référence", font_size=18, color=DIM)).arrange(m.RIGHT, buff=0.3).next_to(ch.axes, m.UP, buff=0.15)
        self.play(m.FadeIn(ch), m.Create(ch.hline(0)), m.FadeIn(lg))
        self.play(m.Create(ch.line(rs, E["nfa_ref"], DIM)), m.Create(ch.line(rs, E["nfa_pic"], RESAMP)), m.FadeIn(dots), run_time=1.6)
        i = int(np.argmin(E["nfa_pic"]))
        self.play(m.Indicate(dots[i], color=ACCENT), m.Write(m.MathTex(rf"r={rs[i]}:\ {E['nfa_pic'][i]:.1f}", font_size=26, color=ACCENT).next_to(dots[i], m.DOWN, buff=0.15)))
        side = m.VGroup(
            m.MathTex(r"r \uparrow\ \Rightarrow\ p \downarrow", font_size=30),
            m.Text("un vote au hasard devient plus rare", font_size=19, color=OK),
            m.MathTex(r"r \uparrow\ \Rightarrow\ N - 2r \downarrow", font_size=30),
            m.Text("moins de distances testables, et la\nfenêtre englobe les autres harmoniques", font_size=19, color=BAD),
            m.MathTex(rf"k(100):\ {E['k'][0]}\ \to\ {E['k'][-1]}\ \text{{votes}}", font_size=28),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).next_to(ch, m.RIGHT, buff=0.55)
        self.play(m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.LEFT) for s in side], lag_ratio=0.35), run_time=2.4)
        self.answer("Optimum vers r = 40 ici ; au-delà, la fenêtre devient trop large par rapport à l'écart entre harmoniques.")

    def e3_kernels(self) -> None:
        self.ask("E3", "Tous les noyaux d'interpolation laissent-ils une trace ?", 4, "L = 8, r = 20")
        E = self.R["E3"]
        names = list(E.keys())
        colors = [ORIG, RESAMP, OK, m.PURPLE]
        ch = Chart((0, 4, 1), (-30, 0, 10), 9.0, 3.3, "", r"\log_{10}\mathrm{NFA}(100)", x_numbers=False).shift(0.3 * m.UP + 0.4 * m.RIGHT)
        bars, labels = m.VGroup(), m.VGroup()
        for i, (n, c) in enumerate(zip(names, colors)):
            top, bot = ch.p(i + 0.5, 0), ch.p(i + 0.5, E[n])
            bar = m.Rectangle(width=1.4, height=abs(top[1] - bot[1]), stroke_width=0).set_fill(c, 0.85).move_to((top + bot) / 2)
            bars.add(bar)
            labels.add(m.VGroup(m.Text(n, font_size=21, color=c).next_to(top, m.UP, buff=0.15), m.MathTex(f"{E[n]:.1f}", font_size=28).next_to(bar, m.DOWN, buff=0.1)))
        self.play(m.FadeIn(ch))
        self.play(m.LaggedStart(*[m.GrowFromEdge(b, m.UP) for b in bars], lag_ratio=0.25), m.LaggedStart(*[m.FadeIn(lb) for lb in labels], lag_ratio=0.25), run_time=2.2)
        self.answer("Oui, tous sont détectés.  Plus le noyau ressemble à un passe-bas parfait (sinc), moins il crée\n"
                    "de répliques : Lanczos ≈ sinc est le plus difficile, le plus proche voisin le plus facile.")

    def e4_factor(self) -> None:
        self.ask("E4", "Et si l'image est peu, ou beaucoup, agrandie ?", 4, "facteur N/M de 0,5 à 10")
        E = self.R["E4"]
        ch = Chart((0, 10.2, 1), (-200, 0, 50), 11.0, 3.6, "N/M", r"\log_{10}\min_d \mathrm{NFA}").shift(0.4 * m.UP + 0.35 * m.RIGHT)
        dots = ch.dots(E["q"], E["nfa"], [OK if d else BAD for d in E["det"]], 0.07)
        one = m.DashedLine(ch.p(1, 0), ch.p(1, -200), color=DIM, stroke_width=2)
        missed = [q for q, d in zip(E["q"], E["det"]) if not d]
        legend = m.VGroup(
            m.Text("● détecté", font_size=20, color=OK),
            m.Text("● non détecté : N/M = " + ", ".join(f"{q:g}".replace(".", ",") for q in missed), font_size=20, color=BAD),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(ch.p(10.2, 0), m.DOWN + m.LEFT, buff=0.3)
        self.play(m.FadeIn(ch), m.Create(ch.hline(0)), m.Create(one))
        self.play(m.Create(ch.line(E["q"], E["nfa"], ORIG, 2)), m.LaggedStart(*[m.FadeIn(d) for d in dots], lag_ratio=0.04), run_time=2.4)
        self.play(m.FadeIn(legend))
        self.answer("N/M = 1 (rien n'a été fait) n'est pas détecté, comme attendu.  Plus N est grand, plus il y a\n"
                    "de patches qui votent (n = N/L) et plus la preuve est forte.  La réduction (N/M < 1) est le cas dur.")

    def e5_r_opt(self) -> None:
        self.ask("E5", "Un seul r convient-il à toutes les distances ?", 4, "on cherche le meilleur r pour chaque d (L = 10, noyau linéaire)")
        E = self.R["E5"]
        ch = Chart((0, 250, 50), (0, 125, 25), 10.0, 3.3, "d", r"r_{\mathrm{opt}}(d)").shift(0.45 * m.UP + 0.3 * m.RIGHT)
        d = [i for i, x in enumerate(E["r_opt"]) if x is not None]
        bound = ch.line(range(250), [min(i, 249 - i) for i in range(250)], DIM, 2)
        cloud = ch.dots(d, [E["r_opt"][i] for i in d], [BAD] * len(d), 0.025).set_opacity(0.5)
        hd = ch.dots([50, 100, 150, 200], [E["r_opt"][h] for h in (50, 100, 150, 200)], [ORIG] * 4, 0.11)
        bl = m.MathTex(r"r \le \min(d,\ N-1-d)", font_size=26, color=DIM).next_to(ch.p(125, 124), m.RIGHT, buff=0.3)
        hl = m.MathTex(rf"\text{{aux harmoniques :}}\ r_{{\mathrm{{opt}}}} = {int(E['r_opt'][100])}", font_size=28, color=ORIG).next_to(ch.p(0, 125), m.RIGHT, buff=0.3)
        self.play(m.FadeIn(ch), m.Create(bound), m.FadeIn(bl))
        self.play(m.FadeIn(cloud), run_time=1.4)
        self.play(m.FadeIn(hd, scale=1.5), m.Write(hl))
        self.answer("Aux vraies distances r_opt ≈ 48 ; ailleurs (points rouges) le « meilleur r » n'est que du bruit.\n"
                    "Le r = 20 du papier est un compromis raisonnable, pas un optimum.")

    # ======================================================================
    # Suite
    # ======================================================================
    def traps(self) -> None:
        self.ask("Partie C", "Les questions pièges du jury", 5)
        qa = [
            ("Deux variables indépendantes ont-elles une corrélation empirique nulle ?", "Non : nulle en espérance, ~1/√L sur un échantillon fini."),
            ("NFA(d) < ε garantit-il zéro faux positif ?", "Non : il borne leur espérance (au plus ε en moyenne)."),
            ("Que se passe-t-il si on change r ?", "p = 1/(2r+1) change, donc toute la binomiale (cf. E2)."),
            ("Pourquoi le sinc est-il indétectable ?", "Passe-bas parfait → aucune réplique → rien à détecter (cf. E3)."),
            ("Pourquoi des patches non chevauchants ?", "Pour l'indépendance, donc la binomiale exacte sous H₀."),
            ("Pourquoi |ρ| et pas ρ signé ?", "On veut « se ressemblent-ils ? », invariant à la phase."),
        ]
        rows = m.VGroup()
        for q, a in qa:
            rows.add(
                m.VGroup(
                    m.Text("• " + q, font_size=20, weight=m.BOLD),
                    m.Text("   → " + a, font_size=19, color=OK),
                ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.06)
            )
        rows.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).next_to(self.cur_title, m.DOWN, buff=0.35)
        if rows.get_bottom()[1] < BOTTOM:
            rows.scale_to_fit_height(self.cur_title.get_bottom()[1] - 0.35 - BOTTOM).next_to(self.cur_title, m.DOWN, buff=0.35)
        self.play(m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in rows], lag_ratio=0.4), run_time=4)
        self.wait(3.5)

    def open_issues(self) -> None:
        self.ask("Vigilance", "Une hypothèse de Q17 est-elle vraiment respectée ?", 5, "l'indépendance des votes — ce que nos tests ont révélé")
        items = [
            ("Plus de fausses alarmes que promis",
             "Sur 200 bruits non rééchantillonnés : 1,6 fausse alarme par signal en moyenne, pour ε = 1."),
            ("Cause probable : la symétrie du spectre",
             "Le spectre d'un signal réel est hermitien : chaque paire de patches existe deux fois et vote deux fois,\n"
             "donc les votes ne sont pas indépendants et la binomiale de Q17 sous-estime la queue."),
            ("Vérification",
             "En ne gardant que les patches n < N/2 : 1,04 fausse alarme par signal, conforme à ε.\n"
             "Les détections faibles (d = 21, 71, 121, 171, NFA ≈ 1) ont probablement la même origine."),
        ]
        blocks = m.VGroup()
        for head, body in items:
            blocks.add(
                m.VGroup(
                    m.Text("▸ " + head, font_size=23, weight=m.BOLD),
                    m.Text(body, font_size=18, line_spacing=1.2, color=m.GREY_A),
                ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            )
        blocks.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35).next_to(self.cur_title, m.DOWN, buff=0.4)
        if blocks.width > 13:
            blocks.scale_to_fit_width(13)
        self.play(m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.RIGHT) for b in blocks], lag_ratio=0.6), run_time=3)
        self.answer("Le contrôle des fausses alarmes repose sur l'indépendance des votes : en 1D, il faut\n"
                    "n'utiliser qu'une moitié du spectre.  À corriger avant le passage en 2D.", color=ACCENT)

    def q8_q9_towards_2d(self) -> None:
        self.ask("Q8 · Q9", "Et en 2D : quels axes, et quelle taille pour ρ ?", 5)
        left = m.VGroup(
            m.Text("Deux axes, deux passes", font_size=24, weight=m.BOLD, color=ACCENT),
            m.Text("une seule fonction « corrélation le long d'un axe » ;\npour l'autre axe, on lui passe le spectre transposé", font_size=19),
            m.MathTex(r"\rho_H,\ \rho_V\ \to\ \mathrm{NFA}_H(d),\ \mathrm{NFA}_V(d)", font_size=30),
            m.Text("→ ce qui permet la validation croisée des axes (III-D)", font_size=19, color=OK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        right = m.VGroup(
            m.Text("Taille de ρ", font_size=24, weight=m.BOLD, color=ACCENT),
            m.MathTex(r"\rho \in \mathbb{R}^{\,n_{\text{patches}} \times n_d}", font_size=32),
            m.Text("~10 × 10 = 100 patches, n_d ≈ N/2 distances", font_size=19),
            m.Text("ex. image 600 × 600 : ρ ≈ 100 × 300", font_size=19),
            m.Text("(code des auteurs : forme transposée)", font_size=17, color=DIM),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        g = m.VGroup(left, right).arrange(m.RIGHT, buff=1.0, aligned_edge=m.UP).next_to(self.cur_title, m.DOWN, buff=0.6)
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in left], lag_ratio=0.3), run_time=2)
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in right], lag_ratio=0.3), run_time=2)
        self.answer("Deux passes séparées → deux matrices ρ (H et V) de taille ~100 × N/2 → deux courbes NFA.")

    def next_steps(self) -> None:
        self.ask("Suite", "Et maintenant ?", 5, "du signal 1D aux vraies images")
        recap = m.VGroup(
            m.Text("Ce qu'il faut retenir :", font_size=22, color=DIM),
            m.Text(
                "on ne juge pas « ρ est-il grand ? » mais « ce nombre de patches d'accord serait-il improbable\n"
                "par hasard ? ».  Le NFA transforme cette improbabilité en un seuil automatique, non supervisé,\n"
                "avec contrôle des fausses alarmes.  Tout cela fonctionne déjà en 1D.",
                font_size=21,
                line_spacing=1.2,
            ),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15)
        steps = m.VGroup(
            *[
                m.Text(s, font_size=22)
                for s in (
                    "1.  Corriger l'indépendance des votes (demi-spectre)",
                    "2.  Passer en 2D : FFT 2D, patches l × l, matrices ρ_H et ρ_V",
                    "3.  Effacer le contenu de l'image : transformée de rang 7×7, débruitage TV",
                    "4.  Écarter les traces JPEG, validation croisée des deux axes",
                    "5.  Accélérer (numpy vectorisé, numba), comparer au code officiel",
                )
            ]
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        thanks = m.Text("Merci !  Vos questions ?   —   Aymen Ouerghi · Mohamed Missaoui", font_size=22, color=ACCENT)
        g = m.VGroup(recap, steps, thanks).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.4).next_to(self.cur_title, m.DOWN, buff=0.35)
        thanks.set_x(0)
        self.play(m.FadeIn(recap))
        self.play(m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.RIGHT) for s in steps], lag_ratio=0.3), run_time=2.5)
        self.play(m.FadeIn(thanks))
        self.wait(3)
