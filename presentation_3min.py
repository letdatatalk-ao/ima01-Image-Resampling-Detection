"""Présentation courte (~3 min) — détecter qu'une image a été agrandie.

Projet IMA01 — Aymen Ouerghi & Mohamed Missaoui

Le fil pédagogique, en images :
  1. une image est un relief ; l'agrandir, c'est tendre une nappe entre les
     colonnes de pixels et la relire plus finement (3D) ;
  2. cette recopie laisse un ÉCHO dans le spectre ;
  3. le PCC reconnaît un écho comme on reconnaît une mélodie jouée plus fort ;
  4. les morceaux du spectre votent : au hasard les jetons s'éparpillent,
     après un agrandissement ils s'empilent en tours (3D) ;
  5. le NFA dit quand une tour est trop haute pour être un hasard.

Les courbes, votes et chiffres viennent de NOTRE code (src/), via le même
cache que avancement_1D.py (animation_cache.json).

À placer dans code_ima/ (à côté de src/ et de avancement_1D.py), puis :
    manimgx preview presentation_3min.py
    manimgx render  presentation_3min.py
    manimgx present presentation_3min.py   # une section = une slide
"""

import math
import sys
from pathlib import Path

import numpy as np

import manimgx as m

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from avancement_1D import ACCENT, BAD, DIM, OK, ORIG, RESAMP, Chart, load_results  # noqa: E402

DEG = m.DEGREES
SAY_WAIT = 3.0  # secondes laissées pour commenter chaque phrase


def caption(text: str, font_size: int = 26, color=m.WHITE) -> m.Text:
    t = m.Text(text, font_size=font_size, color=color, line_spacing=1.15)
    if t.width > 13:
        t.scale_to_fit_width(13)
    return t


def answer_box(text: str, color=OK) -> m.VGroup:
    t = caption(text, 23)
    box = m.RoundedRectangle(
        width=t.width + 0.6, height=t.height + 0.4, corner_radius=0.12, stroke_color=color, stroke_width=2
    ).set_fill(m.BLACK, 0.85)
    box.move_to(t)
    g = m.VGroup(box, t)
    g.to_edge(m.DOWN, buff=0.3)
    return g


class Presentation3min(m.ThreeDScene):
    def construct(self) -> None:
        self.R = load_results()
        self.D = self.R["demo"]
        self.flat()
        self.title()
        self.terrain_3d()
        self.echo()
        self.recognise()
        self.vote_towers_3d()
        self.decide()
        self.recap()
        self.status()

    # -- outils ----------------------------------------------------------------
    def flat(self) -> None:
        """Caméra de face : les parties 2D se lisent comme des diapositives."""
        self.set_camera_orientation(phi=0, theta=-90 * DEG)

    def clear(self, run_time: float = 0.5) -> None:
        if self.mobjects:
            self.play(*[m.FadeOut(x) for x in self.mobjects], run_time=run_time)

    def question(self, text: str, sub: str | None = None, fixed: bool = False) -> m.Text:
        """Une question, posée au centre, qui devient le titre. Pas de numéro : c'est le fil du récit."""
        self.next_section(text)
        self.clear()
        q = m.Text(text, font_size=44, weight=m.BOLD, color=ACCENT)
        if q.width > 12.8:
            q.scale_to_fit_width(12.8)
        grp = m.VGroup(q)
        if sub:
            grp.add(caption(sub, 24, m.GREY_A))
        grp.arrange(m.DOWN, buff=0.3)
        if fixed:
            self.add_fixed_in_frame_mobjects(*grp)
            for g in grp:
                self.remove(g)
        self.play(m.Write(q), *(m.FadeIn(g) for g in grp[1:]), run_time=1.2)
        self.wait(2.2)
        title = m.Text(text, font_size=32, weight=m.BOLD, color=ACCENT).to_edge(m.UP, buff=0.35)
        if title.width > 13:
            title.scale_to_fit_width(13)
        if fixed:
            self.add_fixed_in_frame_mobjects(title)
            self.remove(title)
        self.play(m.ReplacementTransform(q, title), *(m.FadeOut(g) for g in grp[1:]), run_time=0.7)
        return title

    def say(self, text: str, y: float, color=m.WHITE, size: int = 24, fixed: bool = False, old: m.Mobject | None = None) -> m.Text:
        t = caption(text, size, color).move_to([0, y, 0])
        if fixed:
            self.add_fixed_in_frame_mobjects(t)
            self.remove(t)
        if old is not None:
            self.play(m.FadeOut(old, shift=0.2 * m.UP), m.FadeIn(t, shift=0.2 * m.UP), run_time=0.7)
        else:
            self.play(m.FadeIn(t, shift=0.2 * m.UP), run_time=0.7)
        self.wait(SAY_WAIT)
        return t

    def answer(self, text: str, wait: float = 5.5, color=OK, fixed: bool = False) -> m.VGroup:
        box = answer_box(text, color)
        if fixed:
            self.add_fixed_in_frame_mobjects(box)
            self.remove(box)
        self.play(m.FadeIn(box, shift=0.2 * m.UP), run_time=0.6)
        self.wait(wait)
        return box

    # ======================================================================
    def title(self) -> None:
        self.next_section("titre")
        t1 = m.Text("Cette image a-t-elle été agrandie ?", font_size=48, weight=m.BOLD)
        t2 = caption("Détection du rééchantillonnage par corrélation spectrale — notre réimplémentation 1D", 24, ACCENT)
        t3 = caption("Aymen Ouerghi  ·  Mohamed Missaoui  —  Projet IMA01", 24, m.GREY_A)
        g = m.VGroup(t1, t2, t3).arrange(m.DOWN, buff=0.45)
        self.play(m.Write(t1), run_time=1.2)
        self.play(m.FadeIn(t2), m.FadeIn(t3))
        self.wait(1.8)

    # -- 1. l'image comme un relief (3D) -----------------------------------
    def terrain_3d(self) -> None:
        self.question("Agrandir une image, que fait-on vraiment ?", fixed=True)
        rng = np.random.default_rng(3)
        n, cell, H = 6, 0.9, 2.2
        vals = rng.uniform(0.15, 1.0, (n, n))

        def xy(i: float, j: float) -> tuple[float, float]:
            return (j - (n - 1) / 2) * cell, ((n - 1) / 2 - i) * cell

        def height(x: float, y: float) -> float:
            """Interpolation bilinéaire des pixels : la « nappe »."""
            j = x / cell + (n - 1) / 2
            i = (n - 1) / 2 - y / cell
            j = min(max(j, 0), n - 1 - 1e-9)
            i = min(max(i, 0), n - 1 - 1e-9)
            i0, j0 = int(i), int(j)
            ti, tj = i - i0, j - j0
            i1, j1 = min(i0 + 1, n - 1), min(j0 + 1, n - 1)
            v = (1 - ti) * ((1 - tj) * vals[i0, j0] + tj * vals[i0, j1]) + ti * ((1 - tj) * vals[i1, j0] + tj * vals[i1, j1])
            return H * v

        def pillar(x: float, y: float, h: float, w: float, color) -> m.Prism:
            p = m.Prism(dimensions=[w, w, max(h, 0.02)])
            p.set_fill(color, 0.9).set_stroke(m.WHITE, 0.5, 0.4)
            return p.move_to([x, y, max(h, 0.02) / 2])

        base = m.Square(side_length=n * cell + 0.4, stroke_color=DIM, stroke_width=1.5).set_fill(m.GREY_E, 0.4)
        pillars = m.VGroup(
            *[pillar(*xy(i, j), H * vals[i, j], cell * 0.8, m.interpolate_color(m.BLUE_E, m.BLUE_A, vals[i, j])) for i in range(n) for j in range(n)]
        )
        self.move_camera(phi=62 * DEG, theta=-55 * DEG, run_time=0.01)
        self.add(base)
        t = self.say("Une image est un relief : chaque pixel est une colonne, haute comme sa luminosité.", -2.75, fixed=True)
        self.play(m.LaggedStart(*[m.GrowFromEdge(p, m.IN) for p in pillars], lag_ratio=0.02), run_time=2)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(1.5)

        half = (n - 1) / 2 * cell
        sheet = m.Surface(
            lambda u, v: np.array([u, v, height(u, v)]),
            u_range=[-half, half],
            v_range=[-half, half],
            resolution=(30, 30),
            checkerboard_colors=False,
            fill_color=ACCENT,
            fill_opacity=0.55,
            stroke_width=0.3,
            stroke_color=ACCENT,
        )
        t = self.say("Agrandir, c'est d'abord tendre une nappe entre les sommets : l'interpolation.", -2.75, ACCENT, fixed=True, old=t)
        self.play(m.Create(sheet), pillars.animate.set_fill(opacity=0.35), run_time=2.6)
        self.wait(2)

        k = 9
        xs = np.linspace(-half, half, k)
        new = m.VGroup(*[pillar(x, y, height(x, y), cell * 0.32, RESAMP) for x in xs for y in xs])
        t = self.say("Puis on relit la hauteur de la nappe en N × N points : les nouveaux pixels.", -2.75, RESAMP, fixed=True, old=t)
        self.play(m.FadeOut(pillars), m.LaggedStart(*[m.GrowFromEdge(p, m.IN) for p in new], lag_ratio=0.01), run_time=2.2)
        self.play(sheet.animate.set_fill(opacity=0.15), run_time=0.6)
        self.wait(3)
        self.play(m.FadeOut(t), run_time=0.3)
        self.answer("Chaque nouveau pixel est un mélange de quelques anciens : l'information d'origine est recopiée.\n"
                    "Cette recopie est la trace que l'on cherche.", fixed=True)
        self.stop_ambient_camera_rotation()

    # -- 2. l'écho dans le spectre --------------------------------------------
    def echo(self) -> None:
        self.clear()
        self.flat()
        self.question("Où se voit cette recopie ?", "Prenons une ligne de pixels (1D) et regardons ses fréquences.")
        D = self.D
        a = Chart((0, 100, 50), (0, 0.3, 0.3), 3.3, 1.9, "n", r"|\tilde u|").shift(4.45 * m.LEFT + 1.0 * m.UP)
        b = Chart((0, 250, 50), (0, 0.3, 0.3), 7.4, 1.9, "n", r"|\tilde v|").shift(2.6 * m.RIGHT + 1.0 * m.UP)
        la = caption("avant : M = 100", 20, ORIG).next_to(a.axes, m.UP, buff=0.1)
        lb = caption("après agrandissement : N = 250", 20, RESAMP).next_to(b.axes, m.UP, buff=0.1)
        sa = a.stems(range(100), D["absU"], ORIG, 2)
        sb = b.stems(range(250), D["absV"], RESAMP, 1.6)
        self.play(m.FadeIn(a), m.FadeIn(la), m.Create(sa), m.FadeIn(b), m.FadeIn(lb), m.Create(sb), run_time=1.6)
        src = m.SurroundingRectangle(m.VGroup(*sa[40:50]), color=OK, buff=0.06, stroke_width=3)
        boxes = m.VGroup(*[m.SurroundingRectangle(m.VGroup(*sb[s : s + 10]), color=OK, buff=0.06, stroke_width=3) for s in (40, 140)])
        self.play(m.Create(src))
        self.play(m.TransformFromCopy(src, boxes[0]), m.TransformFromCopy(src, boxes[1]), run_time=1.1)
        y = b.p(0, 0)[1] - 0.8
        arrow = m.DoubleArrow([boxes[0].get_center()[0], y, 0], [boxes[1].get_center()[0], y, 0], buff=0, color=OK, stroke_width=3)
        lab = m.MathTex(r"\text{écart}\ d = M \bmod N = 100", font_size=30, color=OK).next_to(arrow, m.DOWN, buff=0.1)
        self.play(m.GrowFromCenter(arrow), m.Write(lab))
        self.wait(3)
        self.answer("L'agrandissement crée un écho dans le spectre : le même motif réapparaît plus loin,\n"
                    "à un écart d = ±M mod N fixé par les tailles avant et après.  C'est le Théorème 1 du papier.")

    # -- 3. reconnaître un écho -------------------------------------------------
    def recognise(self) -> None:
        self.question("Comment reconnaître un écho ?", "Comme une mélodie rejouée plus fort ou décalée : seule la forme compte.")
        V = np.array(self.D["absV"])

        def norm(seg: np.ndarray) -> np.ndarray:
            s = seg - seg.mean()
            return s / (np.abs(s).max() + 1e-12)

        def panel(start: int, shift, name: str, value: float, color) -> m.VGroup:
            ch = Chart((0, 9, 3), (-1, 1, 1), 5.0, 2.4, "", "", x_numbers=False).shift(shift)
            ref = ch.line(range(10), norm(V[40:50]), ORIG, 3.5)
            other = ch.line(range(10), norm(V[start : start + 10]), color, 3.5)
            t = caption(name, 20, color).next_to(ch.axes, m.UP, buff=0.12)
            v = m.MathTex(rf"\mathrm{{PCC}} = {value:.2f}", font_size=36, color=color).next_to(ch, m.DOWN, buff=0.2)
            return m.VGroup(ch, ref, other, t, v)

        p1 = panel(140, 3.4 * m.LEFT + 0.55 * m.UP, "le motif (bleu) et son écho à d = 100", self.D["pcc_copy"], OK)
        p2 = panel(77, 3.4 * m.RIGHT + 0.55 * m.UP, "le motif et un morceau quelconque (d = 37)", self.D["pcc_other"], DIM)
        self.play(m.FadeIn(p1[0]), m.FadeIn(p1[3]), m.Create(p1[1]), m.Create(p1[2]), run_time=1.3)
        self.play(m.Write(p1[4]))
        self.play(m.FadeIn(p2[0]), m.FadeIn(p2[3]), m.Create(p2[1]), m.Create(p2[2]), run_time=1.3)
        self.play(m.Write(p2[4]))
        self.wait(2.5)
        self.answer("La corrélation de Pearson complexe (PCC) compare les formes en ignorant volume et phase :\n"
                    "≈ 1 pour un écho, plus bas pour deux morceaux sans rapport.")

    # -- 4. les votes empilés (3D) ---------------------------------------------
    def vote_towers_3d(self) -> None:
        self.clear()
        self.question("Un écho, ou une coïncidence ?", "Une seule ressemblance peut être due au hasard.", fixed=True)
        D = self.D
        r = 20
        sx, w, h = 0.046, 0.075, 0.13

        def x_of(d: float) -> float:
            return (d - 125) * sx

        def towers(k: list[float], y: float, color) -> m.VGroup:
            cubes = m.VGroup()
            for d in range(r, 250 - r):
                for s in range(int(k[d])):
                    c = m.Prism(dimensions=[w, 0.5, h * 0.92]).set_fill(color, 0.95).set_stroke(m.WHITE, 0.3, 0.3)
                    cubes.add(c.move_to([x_of(d), y, h * (s + 0.5)]))
            return cubes

        floor = m.VGroup(
            *[
                m.Rectangle(width=x_of(250 - r) - x_of(r) + 0.3, height=0.8, stroke_color=DIM, stroke_width=1).set_fill(m.GREY_E, 0.35).move_to([0, y, 0])
                for y in (1.1, -1.1)
            ]
        )
        ticks = m.VGroup(
            *[m.MathTex(str(d), font_size=26, color=OK).move_to([x_of(d), -1.95, 0]) for d in (50, 100, 150, 200)]
        )
        ref = towers(D["k_ref"], 1.1, DIM)
        res = towers(D["k"], -1.1, RESAMP)
        self.move_camera(phi=62 * DEG, theta=-72 * DEG, run_time=0.01)
        self.add(floor)
        t = self.say("Le spectre est coupé en 25 morceaux.  Chacun vote pour l'écart d où il voit le meilleur écho.", -2.75, fixed=True)
        self.wait(1)
        t = self.say("Signal NON agrandi : les jetons s'éparpillent, 5 au plus sur une même distance.", -2.75, m.GREY_A, fixed=True, old=t)
        self.play(m.LaggedStart(*[m.FadeIn(c, shift=0.6 * m.OUT) for c in ref], lag_ratio=0.004), run_time=2.4)
        self.begin_ambient_camera_rotation(rate=0.07)
        t = self.say("Signal agrandi : les jetons s'empilent en tours aux écarts prévus par la théorie.", -2.75, RESAMP, fixed=True, old=t)
        self.play(m.LaggedStart(*[m.FadeIn(c, shift=0.6 * m.OUT) for c in res], lag_ratio=0.004), run_time=2.8)
        self.add(ticks)
        self.wait(5)
        self.play(m.FadeOut(t), run_time=0.3)
        self.answer("On ne juge pas « une ressemblance est-elle forte ? » mais « autant de morceaux d'accord\n"
                    "sur le même écart, est-ce plausible par hasard ? »  Ici : 20 votes sur 25 à d = 100.", fixed=True)
        self.stop_ambient_camera_rotation()

    # -- 5. le NFA : quand affirmer ? -------------------------------------------
    def decide(self) -> None:
        self.clear()
        self.flat()
        self.question("Quand peut-on affirmer « agrandie » ?", "On calcule à quel point la tour serait improbable par pur hasard.")
        D = self.D
        steps = m.VGroup(
            m.MathTex(r"\text{au hasard, un morceau vote pour } d \text{ avec } p = \tfrac{1}{2r+1} = \tfrac{1}{41}", font_size=32),
            m.MathTex(
                rf"\mathbb{{P}}\big[\,\ge {int(D['k'][100])} \text{{ votes sur }} {D['n_patches']}\,\big] \approx 10^{{{D['p_k100']:.0f}}}",
                font_size=34,
                color=OK,
            ),
            caption("plus improbable que de gagner trois fois de suite au loto", 21, m.GREY_A),
            m.MathTex(
                r"\mathrm{NFA} = \underbrace{210}_{\text{écarts testés}} \times\ \mathbb{P}\ \ll\ 1\ \Rightarrow\ \text{détection}",
                font_size=34,
                color=ACCENT,
            ),
        ).arrange(m.DOWN, buff=0.28).shift(1.2 * m.UP)
        for st in steps:
            self.play(m.FadeIn(st, shift=0.15 * m.UP), run_time=0.8)
            self.wait(1.6)
        self.wait(1.5)
        summary = m.MathTex(
            rf"\text{{par hasard}}:\ \mathbb{{P}} \approx 10^{{{D['p_k100']:.0f}}}"
            r"\qquad \mathrm{NFA} = 210 \times \mathbb{P} \ll 1\ \Rightarrow\ \text{détection}",
            font_size=32,
            color=ACCENT,
        ).next_to(m.UP * 3.0, m.DOWN, buff=0.35)
        self.play(m.ReplacementTransform(steps, summary), run_time=0.9)
        nf = Chart((0, 250, 50), (-30, 5, 10), 9.0, 2.7, "d", r"\log_{10}\mathrm{NFA}").shift(0.25 * m.UP + 1.0 * m.RIGHT)
        xs = [i for i, y in enumerate(D["lognfa"]) if y is not None]
        self.play(m.FadeIn(nf), m.Create(nf.hline(0)))
        self.play(
            m.Create(nf.line(xs, [D["lognfa_ref"][i] for i in xs], DIM, 2, clip=(-30, 5))),
            m.Create(nf.line(xs, [D["lognfa"][i] for i in xs], RESAMP, 2.5, clip=(-30, 5))),
            run_time=1.8,
        )
        lg = m.VGroup(caption("agrandi", 19, RESAMP), caption("non agrandi", 19, DIM), caption("seuil ε = 1", 19, BAD)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        lg.next_to(nf, m.LEFT, buff=0.35)
        self.play(m.FadeIn(lg))
        self.wait(2.5)
        self.answer("Le NFA est le nombre de fausses alarmes attendues.  On ne détecte que si NFA < 1 :\n"
                    "un seuil automatique, sans apprentissage, qui contrôle les erreurs.", wait=2.5)

    # -- récapitulatif : toute la méthode en une image ------------------------
    def recap(self) -> None:
        self.question("Toute la méthode, en une image")
        items = [
            ("relief", "agrandir recopie\nl'information", ORIG),
            ("écho", "la copie laisse un écho\ndans le spectre", RESAMP),
            ("votes", "les morceaux du spectre\nvotent pour l'écart", OK),
            ("seuil", "NFA < 1 : trop improbable\npour être un hasard", ACCENT),
        ]
        cards = m.VGroup()
        for head, body, color in items:
            box = m.RoundedRectangle(width=2.75, height=2.1, corner_radius=0.15, stroke_color=color, stroke_width=3)
            content = m.VGroup(caption(head, 30, color), caption(body.replace("\\n", "\n"), 19, m.GREY_A)).arrange(m.DOWN, buff=0.25)
            content.move_to(box)
            cards.add(m.VGroup(box, content))
        cards.arrange(m.RIGHT, buff=0.55).shift(0.4 * m.UP)
        arrows = m.VGroup(*[m.Arrow(cards[i].get_right(), cards[i + 1].get_left(), buff=0.06, stroke_width=4) for i in range(3)])
        for i, c in enumerate(cards):
            anims: list[m.Animation] = [m.FadeIn(c, shift=0.2 * m.RIGHT)]
            if i:
                anims.append(m.GrowArrow(arrows[i - 1]))
            self.play(*anims, run_time=0.8)
            self.wait(1.8)
        self.answer("On ne se demande pas « cette ressemblance est-elle forte ? » mais « autant d'accord\n"
                    "serait-il plausible par hasard ? » — un seuil automatique, sans apprentissage.", color=ACCENT)

    # -- 6. où en sommes-nous ---------------------------------------------------
    def status(self) -> None:
        self.question("Où en sommes-nous ?")
        E1, E2, E3 = self.R["E1"], self.R["E2"], self.R["E3"]
        iL = int(np.argmin(E1["nfa_pic"]))
        ir = int(np.argmin(E2["nfa_pic"]))
        hard = max(E3, key=E3.get)  # le NFA le plus haut = le plus dur
        done = m.VGroup(
            caption("Fait — tout le pipeline en 1D", 26, OK),
            caption("✓ rééchantillonnage, spectre, PCC, votes, NFA : 5 modules", 20),
            caption("✓ 10 tests unitaires, tous verts", 20),
            caption(f"✓ réglages étudiés : patchs L ≈ {E1['L'][iL]} à 8, voisinage r ≈ {E2['r'][ir]}", 20),
            caption(f"✓ 4 noyaux détectés ({hard} le plus difficile)", 20),
            caption("✓ image non agrandie (N/M = 1) jamais détectée", 20),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        todo = m.VGroup(
            caption("Ensuite — les vraies images", 26, ACCENT),
            caption("→ passer en 2D (deux axes)", 20),
            caption("→ effacer le contenu de l'image (résidu)", 20),
            caption("→ gérer la compression JPEG", 20),
            caption("⚠ à corriger : en 1D le spectre est symétrique,\n   chaque vote compte deux fois (≈ 1,6 fausse\n   alarme au lieu de 1) → n'en garder qu'une moitié", 18, BAD),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        g = m.VGroup(done, todo).arrange(m.RIGHT, buff=0.9, aligned_edge=m.UP).shift(0.35 * m.UP)
        self.play(m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.RIGHT) for x in done], lag_ratio=0.25), run_time=2.2)
        self.play(m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.RIGHT) for x in todo], lag_ratio=0.25), run_time=1.8)
        end = caption("Merci !  Vos questions ?", 30, ACCENT).to_edge(m.DOWN, buff=0.45)
        self.play(m.FadeIn(end))
        self.wait(6)
