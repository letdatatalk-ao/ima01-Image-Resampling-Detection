"""Le pipeline complet, pas à pas — de l'interpolation à la décision, puis l'étude de sensibilité.

Projet IMA01 — Aymen Ouerghi & Mohamed Missaoui

Enchaînement (chaque étape répond à la question laissée par la précédente) :
  1. l'interpolation en 3D : pixels → hauteurs → une copie du noyau φ par pixel →
     leur somme f = Σ u_k φ(x − k) est un relief continu → on le relit en j·M/N ;
  2. sur une ligne (1D), notre resample_1d fait la même chose ;
  3. la recopie laisse un écho dans le spectre ; on découpe le spectre en patches ;
  4. la matrice ρ (25 patches × 250 distances) : ce que vaut une case, une ligne, une colonne ;
  5. le rayon r et les votes : un patch vote pour d si ρ y est un maximum local ;
     compter les votes par colonne donne k(d) ;
  6. la probabilité : sous H₀, k(d) ~ B(25, 1/41) — on attend 0,6 vote ;
  7. la décision : NFA(d) = 210 × P[k ≥ k(d)] < ε ;
  8. le pipeline en tailles : ρ (25×250) → k (250) → NFA (250) → décision ;
  9. l'étude de sensibilité (L, r, noyau, facteur N/M).

Toutes les valeurs (spectres, ρ, votes, NFA, expériences) viennent de NOTRE code (src/),
via le cache de avancement_1D.py. L'interpolation 3D utilise notre cubic_kernel.

À placer dans code_ima/ (à côté de src/, avancement_1D.py et presentation_3min.py), puis :
    manimgx present presentation_pipeline.py   # une section = une slide
    manimgx render  presentation_pipeline.py
"""

import math
import sys
from pathlib import Path

import numpy as np

import manimgx as m

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from avancement_1D import ACCENT, BAD, DIM, OK, ORIG, RESAMP, Chart  # noqa: E402
from presentation_3min import DEG, Presentation3min, caption  # noqa: E402
from src.resampling import cubic_kernel, wrap  # noqa: E402

R_WIN = 20  # rayon r utilisé partout (comme le papier)


def heat_rgb(a: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Une palette simple bleu nuit → bleu → jaune pour la matrice ρ."""
    t = np.clip((a - lo) / (hi - lo), 0, 1)[..., None]
    c0 = np.array([12, 18, 40])
    c1 = np.array([40, 120, 200])
    c2 = np.array([250, 215, 90])
    rgb = np.where(t < 0.5, c0 + (c1 - c0) * (t / 0.5), c1 + (c2 - c1) * ((t - 0.5) / 0.5))
    return rgb.astype(np.uint8)


def as_image(rgb: np.ndarray, rep_y: int, rep_x: int) -> m.ImageMobject:
    big = np.repeat(np.repeat(rgb, rep_y, axis=0), rep_x, axis=1)
    alpha = np.full(big.shape[:2] + (1,), 255, dtype=np.uint8)
    return m.ImageMobject(np.concatenate([big, alpha], axis=2), resampling_algorithm=0)


def local_max_votes(rho: np.ndarray, r: int) -> np.ndarray:
    """La même règle que notre count_votes, mais en gardant QUI vote : V[i, d] = 1 si
    ρ[i, d] est le maximum de ρ[i, d−r … d+r]."""
    n_p, n_d = rho.shape
    V = np.zeros_like(rho)
    for d in range(r, n_d - r):
        V[:, d] = rho[:, d] == rho[:, d - r : d + r + 1].max(axis=1)
    return V


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(e: float) -> str:
    """10 puissance e, écrit avec des exposants Unicode (pour du texte simple)."""
    return "10" + f"{e:.0f}".translate(SUP)


def binom_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


class PresentationPipeline(Presentation3min):
    def construct(self) -> None:
        from avancement_1D import load_results

        self.R = load_results()
        self.D = self.R["demo"]
        self.rho = np.array(self.D["rho"])
        self.flat()
        self.title_slide()
        self.interpolation_3d()
        self.line_1d()
        self.echo()
        self.patches()
        self.rho_matrix()
        self.votes()
        self.vote_towers_3d()
        self.probability()
        self.nfa_decision()
        self.pipeline_sizes()
        self.sensitivity()
        self.status()

    # ======================================================================
    def title_slide(self) -> None:
        self.next_section("titre")
        t1 = m.Text("Cette image a-t-elle été agrandie ?", font_size=48, weight=m.BOLD)
        t2 = caption("De l'interpolation à la décision : le pipeline pas à pas, et notre étude de sensibilité", 24, ACCENT)
        chain = m.MathTex(
            r"\text{interpolation} \to \text{spectre} \to \rho \to k(d) \to \mathcal{B}(n,p) \to \mathrm{NFA}(d) < \varepsilon",
            font_size=32,
        )
        t3 = caption("Aymen Ouerghi  ·  Mohamed Missaoui  —  Projet IMA01", 24, m.GREY_A)
        m.VGroup(t1, t2, chain, t3).arrange(m.DOWN, buff=0.45)
        self.play(m.Write(t1), run_time=1.2)
        self.play(m.FadeIn(t2))
        self.play(m.Write(chain), run_time=1.6)
        self.play(m.FadeIn(t3))
        self.wait(2.5)

    # -- 1. l'interpolation, en 3D --------------------------------------------
    def interpolation_3d(self) -> None:
        self.question("Comment agrandit-on une image ?", "Commençons par l'interpolation : de la grille de pixels à un relief continu.", fixed=True)
        Mg, zs = 6, 1.8
        rng = np.random.default_rng(11)
        u = np.round((0.12 + 0.8 * rng.random((Mg, Mg))) * 255) / 255
        c0 = (Mg - 1) / 2  # le pixel (k, l) est au point (k − c0, l − c0)

        def phi(x):
            return cubic_kernel(wrap(x, Mg))

        def f(x: float, y: float) -> float:
            """Notre interpolation : f(x, y) = Σ u_kl φ(x − k) φ(y − l) (noyau cubique, périodique)."""
            ks = np.arange(Mg)
            return float(phi(x - ks) @ u @ phi(y - ks))

        SC = 0.72  # échelle à l'écran : la grille tient au-dessus des sous-titres

        def P(x: float, y: float, z: float = 0.0) -> np.ndarray:
            return np.array([(x - c0) * SC, (y - c0) * SC + 0.25, z])

        # a) la grille de pixels vue de dessus, avec leurs valeurs
        cells = m.VGroup()
        for k in range(Mg):
            for l in range(Mg):
                g = float(u[k, l])
                sq = m.Square(side_length=0.98 * SC, stroke_width=1, stroke_color=DIM).set_fill(m.ManimColor.from_rgb((g, g, g)), 1)
                sq.move_to(P(k, l))
                lab = m.Text(str(int(round(g * 255))), font_size=16, color=m.BLACK if g > 0.5 else m.WHITE).move_to(P(k, l, 0.01))
                cells.add(m.VGroup(sq, lab))
        t = self.say("Au départ : une grille de pixels.  Chaque pixel k a une valeur u_k (son niveau de gris).", -3.0, fixed=True)
        self.play(m.LaggedStart(*[m.FadeIn(c) for c in cells], lag_ratio=0.01), run_time=1.2)

        # b) on incline : chaque valeur devient une hauteur
        t = self.say("On incline : chaque valeur devient une hauteur — une tige verte de longueur u_k.", -3.0, fixed=True, old=t)
        self.move_camera(phi=58 * DEG, theta=-40 * DEG, run_time=2)
        stems = m.VGroup()
        for k in range(Mg):
            for l in range(Mg):
                top = P(k, l, zs * u[k, l])
                stems.add(m.VGroup(m.Line(P(k, l), top, color=OK, stroke_width=4), m.Dot(top, radius=0.07, color=OK)))
        self.play(
            *[m.FadeOut(c[1]) for c in cells],
            *[c[0].animate.set_fill(opacity=0.55) for c in cells],
            m.LaggedStart(*[m.Create(s_) for s_ in stems], lag_ratio=0.02),
            run_time=2,
        )

        # c) une copie du noyau φ, centrée sur un pixel et multipliée par sa valeur
        k0, l0 = 2, 3
        hero = m.Surface(
            lambda a, b: P(a, b, zs * u[k0, l0] * float(phi(a - k0)) * float(phi(b - l0))),
            u_range=[k0 - 2, k0 + 2],
            v_range=[l0 - 2, l0 + 2],
            resolution=(28, 28),
            checkerboard_colors=False,
            fill_color=RESAMP,
            fill_opacity=0.75,
            stroke_color=RESAMP,
            stroke_width=0.4,
        )
        support = m.DashedVMobject(m.Square(side_length=4 * SC, color=RESAMP).move_to(P(k0, l0)), num_dashes=40)
        t = self.say(
            f"Interpolation : on centre une copie du noyau φ (ici cubique) sur le pixel ({k0}, {l0}), multipliée par sa valeur.\n"
            "Le carré pointillé est le support de φ : ce pixel n'influence que ses voisins proches.",
            -3.0,
            RESAMP,
            size=22,
            fixed=True,
            old=t,
        )
        self.play(m.Create(support), m.Create(hero), run_time=2.2)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(2)

        # d) une copie par pixel
        copies = m.VGroup(
            *[
                m.Surface(
                    lambda a, b, k=k, l=l: P(a, b, zs * u[k, l] * float(phi(a - k)) * float(phi(b - l))),
                    u_range=[max(k - 2, -0.5), min(k + 2, Mg - 0.5)],
                    v_range=[max(l - 2, -0.5), min(l + 2, Mg - 0.5)],
                    resolution=(10, 10),
                    checkerboard_colors=False,
                    fill_color=m.interpolate_color(m.BLUE, m.PINK, (k * Mg + l) / (Mg * Mg)),
                    fill_opacity=0.28,
                    stroke_width=0.2,
                )
                for k in range(Mg)
                for l in range(Mg)
                if (k, l) != (k0, l0)
            ]
        )
        t = self.say("On fait de même pour chacun des 36 pixels : 36 bosses, chacune à la hauteur de son pixel.", -3.0, fixed=True, old=t)
        self.play(m.FadeOut(support), m.LaggedStart(*[m.FadeIn(c) for c in copies], lag_ratio=0.03), run_time=2.5)
        self.wait(1)

        # e) leur somme : le relief continu
        terrain = m.Surface(
            lambda a, b: P(a, b, zs * f(a, b)),
            u_range=[-0.5, Mg - 0.5],
            v_range=[-0.5, Mg - 0.5],
            resolution=(36, 36),
            checkerboard_colors=False,
            fill_color=ACCENT,
            fill_opacity=0.85,
            stroke_color=ACCENT,
            stroke_width=0.3,
        )
        formula = m.MathTex(r"f(x) = \sum_k u_k\, \varphi(x - k)", font_size=40, color=ACCENT).to_corner(m.UR, buff=0.4).shift(0.7 * m.DOWN)
        self.add_fixed_in_frame_mobjects(formula)
        self.remove(formula)
        t = self.say("On additionne toutes les copies : f(x) = Σ u_k φ(x − k).  C'est le relief continu — il passe par les sommets verts.", -3.0, ACCENT, size=22, fixed=True, old=t)
        self.play(m.FadeOut(hero), m.FadeOut(copies), m.FadeIn(terrain), m.Write(formula), run_time=2.5)
        self.wait(2.5)

        # f) le rééchantillonnage : des rayons rouges aux nouvelles positions j·M/N
        Nn = 5
        rays = m.VGroup()
        news = []
        for a in range(Nn):
            for b in range(Nn):
                x, y = a * Mg / Nn, b * Mg / Nn
                z = f(x, y)
                news.append((x, y, z))
                rays.add(m.VGroup(m.Line(P(x, y), P(x, y, zs * z), color=BAD, stroke_width=3.5), m.Dot(P(x, y, zs * z), radius=0.08, color=BAD)))
        f2 = m.MathTex(r"v_j = f\!\left(j\,\tfrac{M}{N}\right)", font_size=40, color=BAD).next_to(formula, m.DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(f2)
        self.remove(f2)
        t = self.say(f"Rééchantillonnage : des rayons rouges partent des nouvelles positions x = j·M/N ({Mg}×{Mg} → {Nn}×{Nn}).\n"
                     "La hauteur où chaque rayon touche le relief est la valeur du nouveau pixel.", -3.0, BAD, size=22, fixed=True, old=t)
        self.play(m.FadeOut(stems), terrain.animate.set_fill(opacity=0.45), m.LaggedStart(*[m.Create(r_) for r_ in rays], lag_ratio=0.05), m.Write(f2), run_time=3)
        self.wait(2.5)

        # g) retour en 2D : la nouvelle grille
        self.stop_ambient_camera_rotation()
        h = Mg / Nn
        new_cells = m.VGroup()
        for x, y, z in news:
            g = float(np.clip(z, 0, 1))
            sq = m.Square(side_length=h * 0.98 * SC, stroke_width=2, stroke_color=BAD).set_fill(m.ManimColor.from_rgb((g, g, g)), 1)
            sq.move_to(P(x, y))
            lab = m.Text(str(int(round(g * 255))), font_size=18, color=m.BLACK if g > 0.5 else m.WHITE).move_to(sq.get_center() + np.array([0, 0, 0.01]))
            new_cells.add(m.VGroup(sq, lab))
        t = self.say("Retour en 2D : la nouvelle grille.  Chaque case contient la hauteur du relief sous son rayon.", -3.0, fixed=True, old=t)
        self.play(m.FadeOut(terrain), m.FadeOut(rays), *[m.FadeOut(c[0]) for c in cells], run_time=0.8)
        self.move_camera(phi=0, theta=-90 * DEG, run_time=1.6)
        self.play(m.LaggedStart(*[m.FadeIn(c) for c in new_cells], lag_ratio=0.03), run_time=1.4)
        self.wait(1.5)
        self.play(m.FadeOut(t), run_time=0.3)
        self.answer("Chaque nouveau pixel est une somme pondérée de quelques anciens (les poids sont les valeurs de φ) :\n"
                    "l'information d'origine est recopiée.  C'est cette recopie que l'on va détecter.", fixed=True)

    # -- 2. la même chose sur une ligne, avec notre code -------------------
    def line_1d(self) -> None:
        self.clear()
        self.flat()
        self.question("Et dans notre code ?", "Sur une ligne de pixels (1D), resample_1d fait exactement cela.")
        D = self.D
        ch = Chart((0, 20, 5), (-3, 3, 3), 11, 2.4, "x", "").shift(0.6 * m.UP + 0.3 * m.RIGHT)
        su = ch.stems(range(20), D["u"], ORIG, 6)
        sv = ch.stems([j * 0.4 for j in range(50)], D["v"], RESAMP, 2.5)
        e = m.MathTex(r"v_j = \sum_k u_k\,\varphi\!\left(j\tfrac{M}{N} - k\right),\quad M = 100 \to N = 250", font_size=32).next_to(ch.axes, m.UP, buff=0.3)
        code = m.Text("v = resample_1d(u, N, kernel)", font_size=22, color=ACCENT, font="monospace").next_to(ch, m.DOWN, buff=0.3)
        self.play(m.FadeIn(ch), m.LaggedStart(*[m.Create(s) for s in su], lag_ratio=0.04), m.Write(e))
        self.play(m.LaggedStart(*[m.Create(s) for s in sv], lag_ratio=0.02), m.FadeIn(code), run_time=1.8)
        self.wait(2)
        self.answer("Même principe : chaque v_j mélange quelques u_k.  Le signal agrandi contient des copies du signal d'origine.\n"
                    "Question suivante : où cette recopie laisse-t-elle une trace mesurable ?")

    # -- 3. découper le spectre en patches ---------------------------------
    def patches(self) -> None:
        self.question("Comment mesurer l'écho partout ?", "On découpe le spectre (pas l'image !) en morceaux : les patches.")
        D = self.D
        ch = Chart((0, 250, 50), (0, 0.3, 0.3), 11, 2.0, "n", r"|\tilde v_n|").shift(0.9 * m.UP + 0.3 * m.RIGHT)
        sb = ch.stems(range(250), D["absV"], RESAMP, 1.6)
        cuts = m.VGroup(*[m.DashedLine(ch.p(10 * i, 0), ch.p(10 * i, 0.3), color=OK, stroke_width=1.5) for i in range(1, 25)])
        self.play(m.FadeIn(ch), m.Create(sb), run_time=1.2)
        self.play(m.LaggedStart(*[m.Create(c) for c in cuts], lag_ratio=0.04), run_time=1.4)
        labs = m.VGroup(*[m.MathTex(f"P_{{{i}}}", font_size=20, color=OK).move_to(ch.p(10 * i + 5, 0.27)) for i in (0, 1, 2, 24)])
        self.play(m.FadeIn(labs))
        info = m.VGroup(
            m.MathTex(r"N = 250 \text{ valeurs},\quad L = 10 \text{ par patch} \ \Rightarrow\ 25 \text{ patches } P_0, \dots, P_{24}", font_size=30),
            caption("Un point du spectre est trop bruité ; un patch de 10 valeurs donne une mesure stable.", 21, m.GREY_A),
        ).arrange(m.DOWN, buff=0.2).next_to(ch, m.DOWN, buff=0.35)
        self.play(m.FadeIn(info))
        self.wait(1.5)
        self.answer("Image/signal → FFT → découpe du spectre en 25 patches.  Il reste à comparer chaque patch\n"
                    "à tous les morceaux situés plus loin : c'est la matrice ρ.")

    # -- 4. la matrice ρ -------------------------------------------------------
    def rho_matrix(self) -> None:
        self.question("Ce que vaut UNE case ρ[i, d]", "« À quel point le patch i ressemble-t-il au morceau situé d positions plus loin ? »")
        D = self.D
        rho = self.rho
        i0 = 4
        ch = Chart((0, 250, 50), (0, 0.3, 0.3), 11, 1.7, "n", r"|\tilde v_n|").shift(1.35 * m.UP + 0.3 * m.RIGHT)
        sb = ch.stems(range(250), D["absV"], RESAMP, 1.4)
        P_box = m.SurroundingRectangle(m.VGroup(*sb[40:50]), color=OK, buff=0.05, stroke_width=3)
        P_lab = m.MathTex("P", font_size=28, color=OK).next_to(P_box, m.UP, buff=0.05)
        dt = m.ValueTracker(0.0)

        def q_box() -> m.Mobject:
            d = int(round(dt.get_value()))
            x0, x1 = ch.p(40 + d - 0.5, 0)[0], ch.p(49 + d + 0.5, 0)[0]
            y0, y1 = ch.p(0, 0)[1] - 0.05, ch.p(0, 0.3)[1]
            r_ = m.Rectangle(width=x1 - x0, height=y1 - y0, color=ACCENT, stroke_width=3).move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
            return m.VGroup(r_, m.MathTex("Q", font_size=28, color=ACCENT).next_to(r_, m.UP, buff=0.05))

        Q = m.always_redraw(q_box)
        readout = m.always_redraw(
            lambda: m.MathTex(
                rf"d = {int(round(dt.get_value()))}\qquad \rho[{i0}, d] = \mathrm{{PCC}}(P, Q) = {rho[i0, int(round(dt.get_value()))]:.2f}",
                font_size=34,
                color=OK if rho[i0, int(round(dt.get_value()))] > 0.8 else m.WHITE,
            ).next_to(ch, m.DOWN, buff=0.35)
        )
        self.play(m.FadeIn(ch), m.Create(sb), run_time=1)
        self.play(m.Create(P_box), m.FadeIn(P_lab), m.FadeIn(Q), m.FadeIn(readout))
        expl = caption("P = les 10 valeurs du patch 4 ;  Q = les 10 valeurs situées d plus loin.", 21, m.GREY_A).next_to(readout, m.DOWN, buff=0.25)
        self.play(m.FadeIn(expl))
        self.play(dt.animate.set_value(37), run_time=2.2)
        self.wait(1)
        self.play(dt.animate.set_value(100), run_time=2.2)
        self.wait(2)
        self.answer("ρ[i, d] ≈ 1 : les deux morceaux ont la même forme (un écho).  ρ[i, d] ≈ 0,2-0,3 : rien à voir.")

        # la grille entière
        self.question("La matrice ρ entière", "Une ligne = un patch,  une colonne = une distance,  une case = une ressemblance entre 0 et 1.")
        img = as_image(heat_rgb(rho), 8, 2)
        img.stretch_to_fit_width(11).stretch_to_fit_height(3.2).shift(0.5 * m.UP + 0.4 * m.RIGHT)
        L, Rr, T, B = img.get_left()[0], img.get_right()[0], img.get_top()[1], img.get_bottom()[1]

        def xd(d: float) -> float:
            return L + (d + 0.5) / 250 * (Rr - L)

        frame = m.Rectangle(width=Rr - L, height=T - B, stroke_color=DIM, stroke_width=1.5).move_to(img)
        ylab = caption("25 patches\n(lignes i)", 20, m.GREY_A).next_to(frame, m.LEFT, buff=0.25)
        xt = m.VGroup(*[m.MathTex(str(d), font_size=22).move_to([xd(d), B - 0.22, 0]) for d in (0, 50, 100, 150, 200, 249)])
        xlab = m.MathTex(r"\text{distances } d \ (250 \text{ colonnes})", font_size=26, color=m.GREY_A).next_to(xt, m.DOWN, buff=0.12)
        dim = m.MathTex(r"\rho \in \mathbb{R}^{\,25 \times 250}", font_size=34, color=ACCENT).next_to(frame, m.UP, buff=0.15).align_to(frame, m.RIGHT)
        self.play(m.FadeIn(img), m.Create(frame), m.FadeIn(ylab), m.FadeIn(xt), m.FadeIn(xlab), m.Write(dim), run_time=1.6)
        self.wait(1)

        def col_box(d: float, color, w: float = 0.12) -> m.Rectangle:
            return m.Rectangle(width=w, height=T - B + 0.12, stroke_color=color, stroke_width=3).move_to([xd(d), (T + B) / 2, 0])

        notes = [
            (0, DIM, "colonne d = 0 : toujours 1 (un patch comparé à lui-même) — on l'ignore"),
            (137, ORIG, f"une colonne quelconque (d = 137) : faible, ≈ {np.median(rho[:, 137]):.2f} — le bruit de fond"),
            (100, OK, f"colonne d = 100 = M mod N : forte sur presque toutes les lignes (médiane {np.median(rho[:, 100]):.2f}) — LA TRACE"),
        ]
        cap = None
        for d, color, text in notes:
            box = col_box(d, color)
            c = caption(text, 21, color).next_to(xlab, m.DOWN, buff=0.2)
            anims: list[m.Animation] = [m.Create(box), m.FadeIn(c)]
            if cap is not None:
                anims.append(m.FadeOut(cap))
            self.play(*anims, run_time=0.8)
            self.wait(2.2)
            cap = c
        self.play(m.FadeOut(cap), run_time=0.3)
        harm = m.VGroup(*[col_box(d, OK, 0.08) for d in (50, 150, 200)])
        hc = caption("et aussi aux harmoniques 50, 150, 200 (= ±kM mod N)", 21, OK).next_to(xlab, m.DOWN, buff=0.2)
        self.play(m.Create(harm), m.FadeIn(hc))
        self.wait(2)

    # -- 5. le rayon r et les votes -------------------------------------------
    def votes(self) -> None:
        self.question("Comment passer de ρ à une décision ?", "On ne demande pas « ρ est-il grand ? » (un seuil fixe échoue) mais « est-ce un sommet local ? »")
        D = self.D
        rho = self.rho
        r = R_WIN
        win = list(range(80, 121))
        one = Chart((80, 120, 10), (0, 1, 0.5), 6.0, 2.3, "e", r"\rho[4, e]").shift(2.9 * m.LEFT + 0.8 * m.UP)
        cur = one.line(win, [rho[4, x] for x in win], ORIG, 2.5)
        br = m.BraceBetweenPoints(one.p(80, 1.05), one.p(120, 1.05), direction=m.UP, color=ACCENT)
        brl = m.MathTex(r"[d - r,\ d + r],\ \ r = 20\ \Rightarrow\ 2r+1 = 41 \text{ distances}", font_size=26, color=ACCENT).next_to(br, m.UP, buff=0.08)
        top = m.Dot(one.p(100, rho[4, 100]), color=OK, radius=0.09)
        self.play(m.FadeIn(one), m.Create(cur), run_time=1.2)
        self.play(m.GrowFromCenter(br), m.FadeIn(brl))
        self.play(m.FadeIn(top, scale=1.6))
        rule = m.VGroup(
            caption("Pour le patch 4 et d = 100 :", 22),
            caption("ρ[4, 100] est la plus grande\ndes 41 valeurs de sa fenêtre", 21, m.GREY_A),
            caption("→ le patch 4 « vote » pour d = 100", 23, OK),
            m.MathTex(r"k(d) = \#\{\, i : \rho[i,d] = \max_{|e-d| \le r} \rho[i,e] \,\}", font_size=30),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(one, m.RIGHT, buff=0.6)
        self.play(m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.LEFT) for x in rule], lag_ratio=0.4), run_time=2.4)
        self.wait(2)
        self.answer("Le rayon r définit la fenêtre où l'on cherche le maximum.  Un vote = « pour ce patch,\n"
                    "la distance d est un sommet local de sa ligne ».")

        # tous les votes, puis le décompte par colonne
        self.question("Compter les votes, colonne par colonne", "On balaye toute la matrice : chaque case est-elle un sommet local de sa ligne ?")
        V = local_max_votes(rho, r)
        rgb = np.where(V[..., None] > 0, np.array([250, 140, 60], dtype=np.uint8), np.array([20, 24, 34], dtype=np.uint8))
        img = as_image(rgb, 8, 2)
        img.stretch_to_fit_width(11).stretch_to_fit_height(2.2).shift(1.45 * m.UP + 0.4 * m.RIGHT)
        L, Rr = img.get_left()[0], img.get_right()[0]
        frame = m.Rectangle(width=Rr - L, height=img.height, stroke_color=DIM, stroke_width=1.5).move_to(img)
        lab = caption("votes : une case orange = « ce patch vote pour cette distance »", 20, RESAMP).next_to(frame, m.UP, buff=0.12)
        self.play(m.FadeIn(img), m.Create(frame), m.FadeIn(lab), run_time=1.2)
        arrow = m.MathTex(r"\Big\downarrow\ \ \text{compter par colonne}", font_size=28, color=ACCENT).next_to(frame, m.DOWN, buff=0.12)
        kk = D["k"]
        bars_ch = Chart((0, 250, 50), (0, 25, 10), 11, 1.7, "d", "k(d)").next_to(arrow, m.DOWN, buff=0.15).align_to(frame, m.RIGHT)
        bars = bars_ch.stems(range(r, 250 - r), kk[r : 250 - r], RESAMP, 3)
        self.play(m.FadeIn(arrow), m.FadeIn(bars_ch))
        self.play(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.004), run_time=2)
        k100 = m.MathTex(rf"k(100) = {int(kk[100])}", font_size=28, color=OK).next_to(bars_ch.p(100, kk[100]), m.RIGHT, buff=0.1)
        k137 = m.MathTex(rf"k(137) = {int(kk[137])}", font_size=24, color=DIM).next_to(bars_ch.p(125, 12), m.UP, buff=0.05)
        self.play(m.Write(k100), m.Write(k137))
        self.wait(2.5)

    # -- 6. la probabilité ------------------------------------------------------
    def probability(self) -> None:
        D = self.D
        n, r = D["n_patches"], R_WIN
        p = 1 / (2 * r + 1)
        k100 = int(D["k"][100])
        self.clear()
        self.flat()
        self.question(f"{k100} votes sur {n} : est-ce beaucoup ?", "Ça dépend de ce que le hasard produirait (hypothèse H₀ : pas d'agrandissement).")
        left = m.VGroup(
            caption("Sous H₀, aucune des 41 distances\nde la fenêtre n'est privilégiée :", 21),
            m.MathTex(r"p = \frac{1}{2r+1} = \frac{1}{41} \approx 0{,}024", font_size=34, color=ACCENT),
            caption("25 patches indépendants :", 21),
            m.MathTex(rf"k \sim \mathcal{{B}}\!\left({n},\ \tfrac{{1}}{{41}}\right),\quad \mathbb{{E}}[k] = np \approx {n * p:.1f}", font_size=32),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).to_edge(m.LEFT, buff=0.5).shift(0.5 * m.UP)
        self.play(m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.RIGHT) for x in left], lag_ratio=0.4), run_time=2.4)
        # la loi binomiale, et la valeur observée, très loin à droite
        ch = Chart((0, 22, 5), (0, 0.6, 0.3), 6.2, 2.6, "k", r"\mathbb{P}_{H_0}(k)").next_to(left, m.RIGHT, buff=0.7)
        pmf = [math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(0, 22)]
        bars = m.VGroup(
            *[
                m.Rectangle(width=0.2, height=max(0.005, ch.p(0, v)[1] - ch.p(0, 0)[1]), stroke_width=0).set_fill(ORIG, 0.9).move_to((ch.p(j, 0) + ch.p(j, v)) / 2)
                for j, v in enumerate(pmf)
            ]
        )
        self.play(m.FadeIn(ch), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.03), run_time=1.6)
        obs = m.Arrow(ch.p(k100, 0.45), ch.p(k100, 0.02), color=OK, buff=0, stroke_width=4)
        obl = m.MathTex(rf"k(100) = {k100}", font_size=28, color=OK).next_to(obs, m.UP, buff=0.08)
        self.play(m.GrowArrow(obs), m.Write(obl))
        tail = m.MathTex(
            rf"\mathbb{{P}}(k \ge {k100}) = \sum_{{j={k100}}}^{{{n}}} \binom{{{n}}}{{j}} p^j (1-p)^{{{n}-j}} \approx 10^{{{D['p_k100']:.0f}}}",
            font_size=34,
            color=OK,
        ).next_to(m.VGroup(left, ch), m.DOWN, buff=0.4)
        self.play(m.Write(tail), run_time=1.6)
        self.wait(2)
        self.answer(f"Le hasard donne ≈ {n * p:.1f} vote à une distance ; on en observe {k100}.  La probabilité d'un tel résultat\n"
                    f"par hasard est ≈ {pow10(D['p_k100'])} : quasi impossible.  C'est elle qui mesure « à quel point c'est anormal ».")

    # -- 7. la décision ----------------------------------------------------------
    def nfa_decision(self) -> None:
        D = self.D
        n, r = D["n_patches"], R_WIN
        p = 1 / (2 * r + 1)
        n_tests = 250 - 2 * r
        self.question("Et la décision ?", f"On a testé {n_tests} distances : plus on teste, plus le hasard a d'occasions de nous piéger.")
        f = m.MathTex(
            rf"\mathrm{{NFA}}(d) = \underbrace{{{n_tests}}}_{{\text{{tests}}}} \times\ \mathbb{{P}}\big(k \ge k(d)\big)\qquad \text{{détection si}}\ \boxed{{\mathrm{{NFA}}(d) < \varepsilon = 1}}",
            font_size=36,
        ).shift(2.0 * m.UP)
        self.play(m.Write(f), run_time=1.6)
        kk = D["k"]
        d_quiet = next(d for d in range(130, 230) if int(kk[d]) == 1)
        rows = []
        for d, color in ((100, OK), (d_quiet, BAD)):
            k = int(kk[d])
            P = binom_tail(k, n, p)
            nfa = n_tests * P
            ok = nfa < 1
            P_s = f"10^{{{math.log10(P):.0f}}}" if P < 1e-3 else f"{P:.2f}".replace(".", "{,}")
            N_s = f"10^{{{math.log10(nfa):.0f}}}" if nfa < 1e-3 else f"{nfa:.0f}"
            rows.append(
                m.MathTex(
                    rf"d = {d}:\ \ k = {k},\ \ \mathbb{{P}} \approx {P_s},\ \ \mathrm{{NFA}} \approx {N_s}\ "
                    + (r"< 1\ \Rightarrow\ \text{détecté}" if ok else r"> 1\ \Rightarrow\ \text{pas détecté}"),
                    font_size=32,
                    color=color,
                )
            )
        tbl = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.25).next_to(f, m.DOWN, buff=0.45)
        for row in rows:
            self.play(m.FadeIn(row, shift=0.15 * m.RIGHT), run_time=0.8)
            self.wait(1.5)
        nf = Chart((0, 250, 50), (-30, 5, 15), 8.6, 1.45, "d", r"\log_{10}\mathrm{NFA}").next_to(tbl, m.DOWN, buff=0.3).shift(0.6 * m.RIGHT)
        xs = [i for i, y in enumerate(D["lognfa"]) if y is not None]
        self.play(m.FadeIn(nf), m.Create(nf.hline(0)))
        self.play(m.Create(nf.line(xs, [D["lognfa"][i] for i in xs], RESAMP, 2.5, clip=(-30, 5))), run_time=1.5)
        self.wait(1.5)
        self.answer("Le NFA = le nombre attendu de fausses alarmes.  On détecte d si NFA(d) < ε = 1 (« moins d'une\n"
                    "fausse alarme en moyenne »).  L'image est déclarée agrandie dès qu'une distance passe sous ε.")

    # -- 8. le pipeline en tailles ----------------------------------------------
    def pipeline_sizes(self) -> None:
        self.question("Le pipeline complet, en tailles")
        items = m.VGroup(
            m.MathTex(r"\rho", font_size=48, color=ORIG),
            m.MathTex(r"k(d)", font_size=44, color=RESAMP),
            m.MathTex(r"\mathbb{P}\big(k \ge k(d)\big)", font_size=40, color=OK),
            m.MathTex(r"\mathrm{NFA}(d)", font_size=44, color=ACCENT),
            m.Text("décision", font_size=30, color=BAD),
        ).arrange(m.RIGHT, buff=1.55).shift(0.9 * m.UP)
        if items.width > 13.2:
            items.scale_to_fit_width(13.2)
        sizes = m.VGroup(
            *[
                caption(t, 19, c).next_to(it, m.UP, buff=0.25)
                for t, c, it in zip(["25 × 250", "vecteur 250", "vecteur 250", "vecteur 250", "NFA < ε"], [ORIG, RESAMP, OK, ACCENT, BAD], items)
            ]
        )
        arrows = m.VGroup(*[m.Arrow(items[i].get_right(), items[i + 1].get_left(), buff=0.12, stroke_width=3) for i in range(4)])
        hows = m.VGroup(
            *[
                caption(t, 17, m.GREY_A).next_to(arrows[i], m.DOWN, buff=0.18)
                for i, t in enumerate(["max local (r)\npuis compter", "binomiale\nB(n, p)", f"× {250 - 2 * R_WIN}\ntests", "seuil\nε = 1"])
            ]
        )
        self.play(m.FadeIn(items[0]), m.FadeIn(sizes[0]))
        for i in range(4):
            self.play(m.GrowArrow(arrows[i]), m.FadeIn(hows[i]), m.FadeIn(items[i + 1], shift=0.2 * m.RIGHT), m.FadeIn(sizes[i + 1]), run_time=0.9)
            self.wait(1.2)
        self.answer("La matrice ρ (25×250) se réduit à un vecteur de votes k, puis à un vecteur de NFA ;\n"
                    "on détecte les distances dont le NFA est sous le seuil — sans apprentissage.", color=ACCENT)

    # -- 9. l'étude de sensibilité ----------------------------------------------
    def sensitivity(self) -> None:
        self.question("De quoi dépend la détection ?", "Notre étude de sensibilité, avec notre code (M = 100, N = 250)")
        E1, E2, E3, E4 = (self.R[k] for k in ("E1", "E2", "E3", "E4"))
        W, H = 4.6, 1.55
        # (a) taille de patch L
        a = Chart((0, 52, 10), (-35, 5, 20), W, H, "L", r"\log\mathrm{NFA}", font_size=18)
        a.add(a.hline(0), a.line(E1["L"], E1["nfa_ref"], DIM, 2), a.line(E1["L"], E1["nfa_pic"], RESAMP, 2.5))
        iL = int(np.argmin(E1["nfa_pic"]))
        a.add(m.Dot(a.p(E1["L"][iL], E1["nfa_pic"][iL]), color=ACCENT, radius=0.07))
        ta = caption(f"Taille de patch L : optimum L ≈ {E1['L'][iL]} à 8\n(L petit : fond ρ élevé ; L grand : peu de votants)", 16)
        # (b) rayon r
        b = Chart((0, 85, 20), (-40, 5, 20), W, H, "r", r"\log\mathrm{NFA}", font_size=18)
        b.add(b.hline(0), b.line(E2["r"], E2["nfa_ref"], DIM, 2), b.line(E2["r"], E2["nfa_pic"], RESAMP, 2.5))
        ir = int(np.argmin(E2["nfa_pic"]))
        b.add(m.Dot(b.p(E2["r"][ir], E2["nfa_pic"][ir]), color=ACCENT, radius=0.07))
        tb = caption(f"Rayon r : optimum r ≈ {E2['r'][ir]}\n(p = 1/(2r+1) baisse, mais moins de distances testables)", 16)
        # (c) noyaux
        names = list(E3.keys())
        c = Chart((0, 4, 1), (-30, 0, 15), W, H, "", r"\log\mathrm{NFA}", font_size=18, x_numbers=False)
        cols = [ORIG, RESAMP, OK, m.PURPLE]
        for i, (nm, col) in enumerate(zip(names, cols)):
            top, bot = c.p(i + 0.5, 0), c.p(i + 0.5, E3[nm])
            c.add(m.Rectangle(width=0.7, height=abs(top[1] - bot[1]), stroke_width=0).set_fill(col, 0.9).move_to((top + bot) / 2))
            c.add(caption(nm.replace("plus proche voisin", "p. proche v."), 13, col).next_to(top, m.UP, buff=0.06))
        tc = caption("Noyau : tous détectés ; Lanczos (≈ sinc,\npasse-bas presque parfait) est le plus difficile", 16)
        # (d) facteur N/M
        d = Chart((0, 10.2, 2), (-200, 0, 100), W, H, "N/M", r"\log\min\mathrm{NFA}", font_size=18)
        d.add(d.hline(0), d.line(E4["q"], E4["nfa"], ORIG, 2), d.dots(E4["q"], E4["nfa"], [OK if x else BAD for x in E4["det"]], 0.045))
        td = caption("Facteur N/M : N/M = 1 jamais détecté (voulu) ;\nplus N grandit, plus la preuve est forte", 16)
        panels = m.VGroup(*[m.VGroup(ch_, t_).arrange(m.DOWN, buff=0.15) for ch_, t_ in ((a, ta), (b, tb), (c, tc), (d, td))])
        panels.arrange_in_grid(rows=2, cols=2, buff=(0.9, 0.35)).next_to(m.UP * 3.0, m.DOWN, buff=0.35)
        if panels.height > 4.55:
            panels.scale_to_fit_height(4.55).next_to(m.UP * 3.0, m.DOWN, buff=0.3)
        lg = m.VGroup(caption("au pic d = 100", 16, RESAMP), caption("meilleur NFA, référence", 16, DIM), caption("seuil ε = 1", 16, BAD)).arrange(m.RIGHT, buff=0.4)
        lg.next_to(panels, m.DOWN, buff=0.15)
        self.play(m.FadeIn(lg))
        for pnl in panels:
            self.play(m.FadeIn(pnl, shift=0.15 * m.UP), run_time=0.8)
            self.wait(2.6)
        self.wait(1)
        self.answer("Les réglages du papier (L petit, r = 20) sont raisonnables ; nos mesures précisent les compromis.\n"
                    "La méthode tient pour tous les noyaux testés et rejette correctement l'absence d'agrandissement.", color=ACCENT)
