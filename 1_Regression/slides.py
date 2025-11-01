from manim import *
from manim_slides import Slide, ThreeDSlide
import random

class MLIntroSlide(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        title = Text("Introduction to Machine Learning", font_size=72, color=BLACK)
        author = Text("by Filip Pawlicki", font_size=48, color=BLACK).next_to(title, DOWN, buff=0.5)

        group = VGroup(title, author).move_to(ORIGIN)

        self.play(FadeIn(group, shift=UP))
        self.next_slide()

        self.play(FadeOut(group))

        title = Text("What is Machine Learning?", font_size=50, color=BLACK).to_edge(UP)
        intro = Text(
            "Machine Learning (ML) is a field of AI that allows computers\n"
            "to learn patterns from data and make predictions or decisions.",
            font_size=32,
            color=BLACK
        ).next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(title, shift=UP))
        self.next_slide()

        self.play(FadeIn(intro, shift=UP))
        self.next_slide()

        layer1 = VGroup(*[Circle(0.2, color=BLACK, fill_opacity=0.1) for _ in range(3)]).arrange(DOWN, buff=0.5).shift(
            LEFT * 2)
        layer2 = VGroup(*[Circle(0.2, color=BLACK, fill_opacity=0.1) for _ in range(4)]).arrange(DOWN, buff=0.5)
        layer3 = VGroup(*[Circle(0.2, color=BLACK, fill_opacity=0.1) for _ in range(2)]).arrange(DOWN, buff=0.5).shift(
            RIGHT * 2)

        neurons = VGroup(layer1, layer2, layer3).move_to(DOWN * 1)

        connections = []
        for l1 in layer1:
            for l2 in layer2:
                connections.append(Line(l1.get_center(), l2.get_center(), stroke_width=1, color=GRAY))
        for l2 in layer2:
            for l3 in layer3:
                connections.append(Line(l2.get_center(), l3.get_center(), stroke_width=1, color=GRAY))

        connections_group = VGroup(*connections)
        network = VGroup(connections_group, neurons)
        self.play(FadeIn(network))

        input_text = Text("", font_size=28, color=BLUE)
        input_text.next_to(layer1, LEFT, buff=1)

        output_text = Text("", font_size=28, color=GREEN)
        output_text.next_to(layer3, RIGHT, buff=1)

        self.add(input_text, output_text)

        examples = [
            ('Who wrote Harry Potter?', 'J.K. Rowling'),
            ('Capital of France?', 'Paris'),
            ('2 + 2 = ?', '4'),
            ('Best science club?', 'Gradient'),
        ]

        for inp, out in examples:
            input_text = Text(f"\"{inp}\"", font_size=28, color=BLUE)
            input_text.next_to(layer1, LEFT, buff=0.5).align_to(layer1, ORIGIN)
            self.play(Write(input_text))

            start = random.choice(layer1)
            middle = random.choice(layer2)
            end = random.choice(layer3)

            path1 = Line(start.get_center(), middle.get_center())
            path2 = Line(middle.get_center(), end.get_center())

            dot = Dot(color=RED, radius=0.07).move_to(path1.get_start())
            self.play(MoveAlongPath(dot, path1), run_time=1, rate_func=linear)
            self.play(MoveAlongPath(dot, path2), run_time=1, rate_func=linear)

            output_text = Text(f"\"{out}\"", font_size=28, color=GREEN)
            output_text.next_to(layer3, RIGHT, buff=0.5).align_to(layer3, ORIGIN)
            self.play(Write(output_text))

            self.play(FadeOut(dot), FadeOut(input_text), FadeOut(output_text))
            self.wait(0.5)

        self.next_slide()

        self.play(FadeOut(network), FadeOut(title), FadeOut(intro), FadeOut(input_text), FadeOut(output_text))

        title = Text("AI, ML & DL", font_size=50, color=BLACK).to_edge(UP)
        self.play(FadeIn(title))
        self.next_slide()

        ai_circle = Circle(radius=2.80, color=BLUE, fill_opacity=0.4)
        ml_circle = Circle(radius=2.0, color=GREEN, fill_opacity=0.2)
        dl_circle = Circle(radius=1.0, color=RED, fill_opacity=0.2)

        ml_circle.move_to(ai_circle.get_center())
        dl_circle.move_to(ai_circle.get_center())

        ai_label = Text("AI", font_size=32, color=BLUE).move_to(ai_circle.get_center() + LEFT * 2.4)
        ml_label = Text("ML", font_size=32, color=GREEN).move_to(ml_circle.get_center() + LEFT * 1.5)
        dl_label = Text("DL", font_size=32, color=RED).move_to(dl_circle.get_center())

        diagram = VGroup(ai_circle, ml_circle, dl_circle, ai_label, ml_label, dl_label).scale(1.0)
        self.play(FadeIn(diagram, shift=UP))
        self.next_slide()

        self.play(FadeOut(diagram), FadeOut(title))

        title2 = Text("ML vs Traditional Programming", font_size=46, color=BLACK).to_edge(UP)

        trad_title = Text("Traditional Programming", font_size=30, color=BLACK).next_to(title2, DOWN, buff=0.5)

        data = Text("Data", font_size=28, color=BLUE)
        rules = Text("Rules", font_size=28, color=PURPLE)
        program = Text("Program", font_size=28, color=BLACK, t2c={"Program": PURPLE})
        answer = Text("Answer", font_size=28, color=GREEN)

        arrow_data = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)
        arrow_rules = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)
        arrow_output = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)

        inputs_group = VGroup(
            VGroup(data, arrow_data).arrange(RIGHT, buff=0.2),
            VGroup(rules, arrow_rules).arrange(RIGHT, buff=0.2)
        ).arrange(DOWN, buff=0.5)

        program_box = Rectangle(width=2.4, height=1, color=BLACK)
        program_label = Text("Program", font_size=26, color=BLACK)
        program_group = VGroup(program_box, program_label)

        inputs_group.next_to(program_group, LEFT, buff=1)
        arrow_output.next_to(program_group, RIGHT, buff=0.2)
        answer.next_to(arrow_output, RIGHT, buff=0.2)

        trad_group = VGroup(inputs_group, program_group, arrow_output, answer).arrange(RIGHT, buff=0.5)
        trad_group.next_to(trad_title, DOWN, buff=0.5)

        ml_title = Text("Machine Learning", font_size=30, color=BLACK).next_to(trad_group, DOWN, buff=1)

        ml_data = Text("Data", font_size=28, color=BLUE)
        ml_answers = Text("Answers", font_size=28, color=GREEN)
        ml_model_box = Rectangle(width=2.4, height=1, color=BLACK)
        ml_model_label = Text("Model", font_size=26, color=BLACK)
        ml_model_group = VGroup(ml_model_box, ml_model_label)
        ml_rules = Text("Rules", font_size=28, color=PURPLE)

        ml_arrow_data = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)
        ml_arrow_answers = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)
        ml_arrow_output = Arrow(start=LEFT, end=RIGHT, stroke_width=3, color=BLACK)

        ml_inputs = VGroup(
            VGroup(ml_data, ml_arrow_data).arrange(RIGHT, buff=0.2),
            VGroup(ml_answers, ml_arrow_answers).arrange(RIGHT, buff=0.2)
        ).arrange(DOWN, buff=0.5)

        ml_inputs.next_to(ml_model_group, LEFT, buff=1)
        ml_arrow_output.next_to(ml_model_group, RIGHT, buff=0.2)
        ml_rules.next_to(ml_arrow_output, RIGHT, buff=0.2)

        ml_group = VGroup(ml_inputs, ml_model_group, ml_arrow_output, ml_rules).arrange(RIGHT, buff=0.5)
        ml_group.next_to(ml_title, DOWN, buff=0.5)

        self.play(FadeIn(title2, shift=UP))
        self.next_slide()

        self.play(Write(trad_title))
        self.next_slide()
        self.play(FadeIn(data), FadeIn(rules))
        self.play(GrowArrow(arrow_data), GrowArrow(arrow_rules))
        self.next_slide()
        self.play(FadeIn(program_group))
        self.next_slide()
        self.play(GrowArrow(arrow_output), FadeIn(answer))
        self.next_slide()

        self.play(Write(ml_title))
        self.next_slide()
        self.play(FadeIn(ml_data), FadeIn(ml_answers))
        self.play(GrowArrow(ml_arrow_data), GrowArrow(ml_arrow_answers))
        self.next_slide()
        self.play(FadeIn(ml_model_group))
        self.next_slide()
        self.play(GrowArrow(ml_arrow_output), FadeIn(ml_rules))
        self.next_slide()

        self.play(
            FadeOut(title2),
            FadeOut(trad_title), FadeOut(trad_group),
            FadeOut(ml_title), FadeOut(ml_group)
        )

        title2 = Text("Main ML Paradigms", font_size=50, color=BLUE_E)
        supervised = Text("Supervised Learning: labeled data → e.g. classification, regression", font_size=32, color=BLACK)
        unsupervised = Text("Unsupervised Learning: unlabeled data → e.g. clustering, PCA", font_size=32, color=BLACK)
        rl = Text("Reinforcement Learning: agent learns via environment interaction", font_size=32, color=BLACK)

        paradigms = VGroup(supervised, unsupervised, rl).arrange(DOWN, buff=0.5)
        group2 = Group(title2, paradigms).arrange(DOWN, buff=1)
        self.play(FadeIn(title2, shift=UP))
        self.next_slide()
        for p in paradigms:
            self.play(FadeIn(p, shift=LEFT))
            self.next_slide()

        examples_title = Text("Real Examples", font_size=50, color=GREEN_E).to_edge(UP)

        supervised = Text("- Supervised: spam detection, house price prediction", font_size=32, color=BLACK)
        unsupervised = Text("- Unsupervised: customer segmentation, anomaly detection", font_size=32, color=BLACK)
        reinforcement = Text("- Reinforcement: self-driving cars, AI in games", font_size=32, color=BLACK)

        examples_group = VGroup(supervised, unsupervised, reinforcement).arrange(DOWN, aligned_edge=LEFT,
                                                                                 buff=0.5).next_to(examples_title, DOWN,
                                                                                                   buff=0.7)

        self.play(FadeOut(group2))
        self.play(FadeIn(examples_title, shift=UP))
        self.next_slide()

        self.play(Write(supervised))
        self.next_slide()

        self.play(Write(unsupervised))
        self.next_slide()

        self.play(Write(reinforcement))
        self.next_slide()

        self.play(FadeOut(examples_title), FadeOut(examples_group))

class RegressionSlide(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        title = Text("What is Regression?", font_size=50, color=BLACK).to_edge(UP)
        desc = Text(
            "Regression is a method to predict numerical values from data.",
            font_size=32, color=BLACK
        ).next_to(title, DOWN, buff=0.7)

        self.play(FadeIn(title, shift=UP))
        self.next_slide()

        self.play(FadeIn(desc, shift=UP))
        self.next_slide()

        examples_title = Text("Examples of Regression", font_size=48, color=GREEN)
        examples = Text(
            "- Predicting car MPG\n"
            "- Weather forecasting\n"
            "- Trend analysis",
            font_size=32, color=BLACK
        )
        group_examples = Group(examples_title, examples).arrange(DOWN, buff=0.5).next_to(desc, DOWN, buff=1)

        self.play(FadeIn(group_examples, shift=UP))
        self.next_slide()

        self.play(FadeOut(title), FadeOut(desc), FadeOut(group_examples))

        title2 = Text("Linear Regression Example", font_size=50, color=BLACK).to_edge(UP)

        axes = Axes(
            x_range=[10, 40, 5],
            y_range=[500, 2500, 500],
            x_length=8,
            y_length=5,
            axis_config={"color": BLACK},
        )

        x_label = Text("MPG", font_size=28, color=BLACK).next_to(axes.x_axis, DOWN, buff=0.5)
        y_label = Text("Weight", font_size=28, color=BLACK).next_to(axes.y_axis, LEFT, buff=0.7).rotate(PI / 2)

        x_ticks = VGroup(*[
            Text(str(i), font_size=20, color=BLACK).next_to(axes.c2p(i, 500), DOWN)
            for i in range(10, 40, 5)
        ])

        y_ticks = VGroup(*[
            Text(str(i), font_size=20, color=BLACK).next_to(axes.c2p(10, i), LEFT)
            for i in range(500, 2500, 500)
        ])

        dots = VGroup(
            Dot(axes.c2p(15, 2000), color=DARK_BLUE),
            Dot(axes.c2p(20, 1800), color=DARK_BLUE),
            Dot(axes.c2p(25, 1700), color=DARK_BLUE),
            Dot(axes.c2p(35, 1400), color=DARK_BLUE),
        )

        self.play(FadeIn(title2))
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.play(FadeIn(x_ticks), FadeIn(y_ticks))
        self.play(LaggedStart(*[FadeIn(dot, scale=0.5) for dot in dots], lag_ratio=0.2))
        self.next_slide()

        x_data = np.array([15, 20, 25, 30])
        y_data = np.array([2000, 1800, 1700, 1400])
        a, b = np.polyfit(x_data, y_data, 1)

        line = axes.plot(lambda x: a * x + b, color=RED)
        self.play(Create(line))
        self.next_slide()

        new_mpg = 22
        predicted_weight = a * new_mpg + b

        vertical_line = DashedLine(
            start=axes.c2p(new_mpg, 500),
            end=axes.c2p(new_mpg, predicted_weight),
            color=GRAY
        )
        horizontal_line = DashedLine(
            start=axes.c2p(new_mpg, predicted_weight),
            end=axes.c2p(10, predicted_weight),
            color=GRAY
        )
        point_on_line = Dot(axes.c2p(new_mpg, predicted_weight), color=ORANGE)

        x_value_label = Text(f"{new_mpg}", font_size=24, color=ORANGE).next_to(vertical_line.get_start(), DOWN, buff=0.2)
        y_value_label = Text(f"{int(predicted_weight)}", font_size=24, color=ORANGE).next_to(horizontal_line.get_end(),
                                                                                            LEFT, buff=0.2)

        self.play(FadeIn(point_on_line))
        self.next_slide()

        self.play(Create(horizontal_line))
        self.play(FadeIn(y_value_label))
        self.next_slide()

        self.play(Create(vertical_line))
        self.play(FadeIn(x_value_label))
        self.next_slide()

        self.play(
            FadeOut(title2), FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(dots), FadeOut(line), FadeOut(vertical_line), FadeOut(horizontal_line),
            FadeOut(point_on_line), FadeOut(x_ticks), FadeOut(y_ticks), FadeOut(x_value_label), FadeOut(y_value_label)
        )

        title3 = Text("Regression Formula", font_size=50, color=GREEN).to_edge(UP)
        formula = MathTex(r"\hat{y} = w \cdot x + b", font_size=80, color=BLACK)
        linear_formula = MathTex(r"f(x) = a \cdot x + b", font_size=80, color=BLACK).move_to(formula.get_center())

        self.play(FadeIn(title3))
        self.next_slide()

        self.play(Write(linear_formula))
        self.next_slide()

        self.play(Transform(linear_formula, formula))
        self.next_slide()

        self.play(FadeOut(title3), FadeOut(linear_formula))

        title4 = Text("Training Models", font_size=50, color=BLACK).to_edge(UP)
        self.play(FadeIn(title4))
        self.next_slide()

        dataset_text = Text("Dataset", font_size=36, color=BLUE).move_to(LEFT * 6)
        model_text = Text("Model", font_size=36, color=PURPLE).move_to(LEFT * 2)
        loss_text = Text("Loss", font_size=36, color=RED).move_to(RIGHT * 2)
        optimizer_text = Text("Optimizer", font_size=36, color=ORANGE).move_to(RIGHT * 6)

        arrow1 = Arrow(dataset_text.get_right(), model_text.get_left(), buff=0.2, color=BLACK)
        arrow2 = Arrow(model_text.get_right(), loss_text.get_left(), buff=0.2, color=BLACK)
        arrow3 = Arrow(loss_text.get_right(), optimizer_text.get_left(), buff=0.2, color=BLACK)

        self.play(FadeIn(dataset_text))
        self.next_slide()

        self.play(FadeIn(arrow1), FadeIn(model_text))
        self.next_slide()

        self.play(FadeIn(arrow2), FadeIn(loss_text))
        self.next_slide()

        self.play(FadeIn(arrow3), FadeIn(optimizer_text))
        self.next_slide()

        self.play(
            FadeOut(model_text),
            FadeOut(loss_text),
            FadeOut(optimizer_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
        )
        dataset_large = dataset_text.copy().scale(1.5).move_to(title4.get_bottom() + DOWN * 0.7)

        train_desc = Text("Train set", font_size=32, color=BLACK).next_to(dataset_large, DOWN).shift(LEFT * 3)
        train_perc = Text("≈70%", font_size=28, color=BLACK).next_to(train_desc, DOWN)

        val_desc = Text("Validation set", font_size=32, color=BLACK).next_to(dataset_large, DOWN)
        val_perc = Text("≈20%", font_size=28, color=BLACK).next_to(val_desc, DOWN)

        test_desc = Text("Test set", font_size=32, color=BLACK).next_to(dataset_large, DOWN).shift(RIGHT * 3)
        test_perc = Text("≈10%", font_size=28, color=BLACK).next_to(test_desc, DOWN)

        self.play(Transform(dataset_text, dataset_large))
        self.next_slide()

        self.play(FadeIn(train_desc), FadeIn(train_perc))
        self.next_slide()

        self.play(FadeIn(val_desc), FadeIn(val_perc))
        self.next_slide()

        self.play(FadeIn(test_desc), FadeIn(test_perc))
        self.next_slide()

        self.play(
            FadeOut(train_desc), FadeOut(train_perc),
            FadeOut(val_desc), FadeOut(val_perc),
            FadeOut(test_desc), FadeOut(test_perc)
        )

        dataset_original = dataset_text.copy().scale(1 / 1.5).move_to(LEFT * 6)
        self.play(Transform(dataset_text, dataset_original))

        self.play(
            FadeIn(model_text),
            FadeIn(loss_text),
            FadeIn(optimizer_text),
            FadeIn(arrow1),
            FadeIn(arrow2),
            FadeIn(arrow3),
        )

        self.next_slide()

        self.play(
            FadeOut(dataset_text),
            FadeOut(loss_text),
            FadeOut(optimizer_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
        )

        model_large = model_text.copy().scale(1.5).move_to(title4.get_bottom() + DOWN * 0.7)
        regression_formula = MathTex(r"\hat{y} = w \cdot x + b", font_size=48, color=BLACK).next_to(model_large,
                                                                                                          DOWN)


        input_neurons = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(3)]).arrange(DOWN, center=True, buff=0.5)
        input_neurons.shift(LEFT * 3)

        hidden_neurons = VGroup(*[Circle(radius=0.3, color=GREEN) for _ in range(4)]).arrange(DOWN, center=True,
                                                                                              buff=0.5)

        output_neurons = VGroup(*[Circle(radius=0.3, color=RED) for _ in range(1)]).arrange(DOWN, center=True)
        output_neurons.shift(RIGHT * 3)

        connections = VGroup()
        for inp in input_neurons:
            for hid in hidden_neurons:
                connections.add(Line(inp.get_center(), hid.get_center(), color=BLACK))
        for hid in hidden_neurons:
            for out in output_neurons:
                connections.add(Line(hid.get_center(), out.get_center(), color=BLACK))

        nn_diagram = VGroup(input_neurons, hidden_neurons, output_neurons, connections).next_to(
            regression_formula, DOWN, buff=0.75
        )

        self.play(Transform(model_text, model_large), FadeIn(regression_formula), FadeIn(nn_diagram))
        self.next_slide()

        model_original = model_text.copy().scale(1 / 1.5).move_to(LEFT * 2)
        self.play(Transform(model_text, model_original), FadeOut(regression_formula), FadeOut(nn_diagram))

        self.play(
            FadeIn(dataset_text),
            FadeIn(loss_text),
            FadeIn(optimizer_text),
            FadeIn(arrow1),
            FadeIn(arrow2),
            FadeIn(arrow3),
        )

        self.next_slide()

        self.play(
            FadeOut(dataset_text),
            FadeOut(model_text),
            FadeOut(optimizer_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
        )

        loss_large = loss_text.copy().scale(1.5).move_to(title4.get_bottom() + DOWN * 0.7)
        mse_formula = MathTex(r"MSE = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2", font_size=48, color=BLACK).next_to(
            loss_large, DOWN
        )

        self.play(Transform(loss_text, loss_large), FadeIn(mse_formula))
        self.next_slide()

        labels = [2, 3, 4]
        preds = [2.1, 2.9, 4.2]
        table_rows = VGroup()
        for i, (y, y_hat) in enumerate(zip(labels, preds)):
            row = Text(f"Label: {y}    Prediction: {y_hat}", font_size=32, color=BLACK).next_to(
                mse_formula, DOWN, buff=0.5 + i * 0.6
            )
            table_rows.add(row)
            self.play(FadeIn(row))
            self.next_slide()

        mse_start = MathTex(r"MSE = \frac{1}{3}(", font_size=48, color=BLACK)
        mse_start.move_to(np.array([-5 + mse_start.width / 2, table_rows.get_bottom()[1] - 0.5, 0]))
        self.play(FadeIn(mse_start))
        self.next_slide()

        mse_parts = []
        for i, (y, y_hat) in enumerate(zip(labels, preds)):
            part_text = f"({y}-{y_hat})^2"
            if i == 0:
                part_tex = MathTex(part_text, font_size=36, color=BLACK).next_to(mse_start, RIGHT, buff=0.2)
            else:
                part_tex = MathTex("+ " + part_text, font_size=36, color=BLACK).next_to(mse_parts[-1], RIGHT, buff=0.2)
            mse_parts.append(part_tex)
            self.play(FadeIn(part_tex))
            self.next_slide()

        mse_end = MathTex(")", font_size=48, color=BLACK).next_to(mse_parts[-1], RIGHT, buff=0.2)
        mse_result = MathTex("= 0.0267", font_size=40, color=RED).next_to(mse_end, RIGHT, buff=0.3)
        self.play(FadeIn(mse_end), FadeIn(mse_result))
        self.next_slide()

        loss_original = loss_text.copy().scale(1 / 1.5).move_to(RIGHT * 2)
        all_loss_elements = VGroup(
            mse_formula, mse_start, mse_end, mse_result,
            *table_rows, *mse_parts
        )
        self.play(
            Transform(loss_text, loss_original),
            FadeOut(all_loss_elements)
        )

        self.play(
            FadeIn(dataset_text),
            FadeIn(model_text),
            FadeIn(optimizer_text),
            FadeIn(arrow1),
            FadeIn(arrow2),
            FadeIn(arrow3),
        )

        self.next_slide()

        self.play(
            FadeOut(dataset_text),
            FadeOut(model_text),
            FadeOut(loss_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
        )

        optimizer_large = optimizer_text.copy().scale(1.5).move_to(title4.get_bottom() + DOWN * 0.7)

        gradient_desc = Text("Gradient Descent", font_size=24, color=BLACK).next_to(
            optimizer_large, DOWN, buff=0.3
        )
        grad_formula = MathTex(r"w := w - \eta \nabla_w L, \quad b := b - \eta \nabla_b L", font_size=50,
                               color=BLACK).next_to(
            gradient_desc, DOWN, buff=0.4
        )

        self.play(
            Transform(optimizer_text, optimizer_large),
            FadeIn(grad_formula),
            FadeIn(gradient_desc)
        )
        self.next_slide()

        expanded_formula = MathTex(
            r"\nabla_w L = \frac{1}{n} \sum_i 2 (\hat{y}_i - y_i) x_i, \quad "
            r"\nabla_b L = \frac{1}{n} \sum_i 2 (\hat{y}_i - y_i)",
            font_size=44, color=BLACK
        ).next_to(gradient_desc, DOWN, buff=0.5)

        expanded_formula_2 = MathTex(
            r"\nabla_w L = \frac{1}{n} \sum_i 2 ((w x_i + b) - y_i) x_i, \quad "
            r"\nabla_b L = \frac{1}{n} \sum_i 2 ((w x_i + b) - y_i)",
            font_size=44, color=BLACK
        ).next_to(gradient_desc, DOWN, buff=0.5)

        self.play(Transform(grad_formula, expanded_formula))
        self.next_slide()

        self.play(Transform(grad_formula, expanded_formula_2))
        self.next_slide()

        x_vals = [1, 2]
        labels = [2, 3]
        preds = [2.5, 2.8]
        gradient_steps_w = []
        gradient_steps_b = []

        for i, (x, y, y_hat) in enumerate(zip(x_vals, labels, preds)):
            step_w = MathTex(
                rf"g_{{w,{i + 1}}} = 2*({y_hat}-{y})*{x} = {2 * (y_hat - y) * x:.2f}",
                font_size=36, color=BLACK
            ).next_to(expanded_formula, DOWN, buff=0.6 + i * 0.8).to_edge(LEFT)

            step_b = MathTex(
                rf"g_{{b,{i + 1}}} = 2*({y_hat}-{y}) = {2 * (y_hat - y):.2f}",
                font_size=36, color=BLACK
            ).next_to(expanded_formula, DOWN, buff=0.6 + i * 0.8).to_edge(RIGHT)

            gradient_steps_w.append(step_w)
            gradient_steps_b.append(step_b)

            self.play(FadeIn(step_w), FadeIn(step_b))
            self.next_slide()

        avg_w_val = sum([2 * (y_hat - y) * x for x, y_hat, y in zip(x_vals, preds, labels)]) / len(labels)
        avg_b_val = sum([2 * (y_hat - y) for y_hat, y in zip(preds, labels)]) / len(labels)

        gradient_avg_w = MathTex(
            rf"g_{{w,avg}} = \frac{{{'+'.join([f'{2 * (y_hat - y) * x:.2f}' for x, y_hat, y in zip(x_vals, preds, labels)])}}}{{{len(labels)}}} = {avg_w_val:.3f}",
            font_size=36, color=BLACK
        ).next_to(gradient_steps_w[-1], DOWN, buff=0.8).to_edge(LEFT)

        gradient_avg_b = MathTex(
            rf"g_{{b,avg}} = \frac{{{'+'.join([f'{2 * (y_hat - y):.2f}' for y_hat, y in zip(preds, labels)])}}}{{{len(labels)}}} = {avg_b_val:.3f}",
            font_size=36, color=BLACK
        ).next_to(gradient_steps_b[-1], DOWN, buff=0.8).to_edge(RIGHT)

        self.play(FadeIn(gradient_avg_w), FadeIn(gradient_avg_b))
        self.next_slide()

        optimizer_original = optimizer_text.copy().scale(1 / 1.5).move_to(RIGHT * 6)
        self.play(
            Transform(optimizer_text, optimizer_original),
            FadeOut(expanded_formula),
            FadeOut(expanded_formula_2),
            FadeOut(*gradient_steps_w),
            FadeOut(*gradient_steps_b),
            FadeOut(gradient_avg_w),
            FadeOut(gradient_avg_b),
            FadeOut(gradient_desc),
            FadeOut(grad_formula)
        )

        self.play(
            FadeIn(dataset_text),
            FadeIn(model_text),
            FadeIn(loss_text),
            FadeIn(arrow1),
            FadeIn(arrow2),
            FadeIn(arrow3),
        )

        self.next_slide()

        self.play(
            FadeOut(dataset_text),
            FadeOut(model_text),
            FadeOut(loss_text),
            FadeOut(optimizer_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(arrow3),
        )

        axes = Axes(
            x_range=[0, 7, 1],
            y_range=[0, 4, 1],
            x_length=9,
            y_length=6,
            axis_config={"color": BLACK}
        ).to_edge(UP, buff=1)

        x_label = Text("Parameter w", color=BLACK, font_size=28).next_to(axes.x_axis.get_end(), UP)
        y_label = Text("Loss L", color=BLACK, font_size=28).next_to(axes.y_axis.get_end(), LEFT * 1.2).shift(DOWN * 0.2)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.next_slide()

        loss_func = lambda x: 0.1 * (x - 1.5) ** 2 * (x - 5.5) ** 2 + 0.1 * (x - 5.5) ** 2 + 0.5
        loss_curve = axes.plot(loss_func, color=RED, stroke_width=4, x_range=[0.75, 6.55])
        self.play(Create(loss_curve))
        self.next_slide()

        initial_lr = "\eta"
        lr_formula = MathTex(r"w := w - " + f"{initial_lr}" + r" \cdot \nabla L", color=BLACK).to_edge(DOWN)
        self.play(FadeIn(lr_formula))
        self.next_slide()

        def animate_point_along_curve(dot, lr, steps=1, substeps=20, run_time_per_step=0.3):
            self.play(lr_formula.animate.become(
                MathTex(r"w := w - " + f"{lr}" + r" \cdot \nabla L", color=BLACK).to_edge(DOWN)
            ))

            for _ in range(steps):
                w_start = axes.p2c(dot.get_center())[0]
                grad = (loss_func(w_start + 0.001) - loss_func(w_start)) / 0.001
                w_end = w_start - lr * grad
                path = VMobject()
                points = [axes.c2p(w_start + (w_end - w_start) * t / substeps,
                                   loss_func(w_start + (w_end - w_start) * t / substeps))
                          for t in range(substeps + 1)]
                path.set_points_smoothly(points)
                self.play(MoveAlongPath(dot, path), run_time=run_time_per_step)

        small_lr = 0.05
        large_lr = 1.2
        good_lr = 0.6

        point_small = Dot(axes.c2p(0.8, loss_func(0.8)), color=BLUE, radius=0.12)
        self.play(FadeIn(point_small))
        animate_point_along_curve(point_small, small_lr, steps=12, run_time_per_step=0.3)
        self.next_slide()

        point_large = Dot(axes.c2p(0.8, loss_func(0.8)), color=PURPLE, radius=0.12)
        self.play(FadeIn(point_large))
        animate_point_along_curve(point_large, large_lr, steps=6, run_time_per_step=0.8)
        self.next_slide()

        point_good = Dot(axes.c2p(0.8, loss_func(0.8)), color=GREEN, radius=0.12)
        self.play(FadeIn(point_good))
        animate_point_along_curve(point_good, good_lr, steps=7, run_time_per_step=0.8)
        self.next_slide()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        title5 = Text("Non-linear Regression", font_size=50, color=BLACK).to_edge(UP)
        desc5 = Text(
            "Used when the relationship between input and output is not linear.\n"
            "Fits curves to capture complex patterns.",
            font_size=32, color=BLACK
        ).next_to(title5, DOWN, buff=0.5)

        self.play(FadeIn(title5, shift=UP), FadeIn(desc5, shift=UP))
        self.next_slide()

        axes_nl = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=7,
            y_length=4,
            axis_config={"color": BLACK}
        ).next_to(desc5, DOWN, buff=0.2)

        x_label = Text("x", color=BLACK, font_size=28).next_to(axes_nl.x_axis.get_end(), UP)
        y_label = Text("y", color=BLACK, font_size=28).next_to(axes_nl.y_axis.get_end(), LEFT * 1.2)

        self.play(Create(axes_nl), Write(x_label), Write(y_label))
        self.next_slide()

        np.random.seed(32)
        x_max = 5
        y_max = 5
        num_points = 10
        x_vals = np.random.uniform(0, x_max, num_points)

        true_coeffs = [0.18, -0.7, 2.0]
        y_vals = true_coeffs[0] * x_vals ** 2 + true_coeffs[1] * x_vals + true_coeffs[2]

        y_vals += np.random.normal(0, 0.2, size=num_points)
        y_vals = np.clip(y_vals, 0, y_max)

        data_points = [Dot(axes_nl.c2p(x, y), color=BLUE) for x, y in zip(x_vals, y_vals)]

        self.play(LaggedStartMap(FadeIn, data_points, shift=UP))
        self.next_slide()

        coeffs = np.polyfit(x_vals, y_vals, deg=2)
        poly = np.poly1d(coeffs)
        curve = axes_nl.plot(poly, color=RED, stroke_width=4, x_range=[0, x_max])
        self.play(Create(curve), run_time=1.5)
        self.next_slide()

        formula = MathTex(r"\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2", color=BLACK).next_to(axes_nl, DOWN, buff=0.25)
        self.play(FadeIn(formula, shift=UP))
        self.next_slide()

        self.play(FadeOut(VGroup(title5, desc5, axes_nl, x_label, y_label, *data_points, curve, formula)))

class MultidimensionalRegression(ThreeDSlide):
    def construct(self):
        self.camera.background_color = WHITE

        title6 = Text("Multidimensional Regression", font_size=50, color=BLACK).to_edge(UP)
        desc6 = Text(
            "Used when the output depends on multiple input features.\n"
            "Fits surfaces to capture complex patterns in higher dimensions.",
            font_size=32, color=BLACK
        ).next_to(title6, DOWN, buff=0.25)

        self.add_fixed_in_frame_mobjects(title6, desc6)
        self.play(FadeIn(title6, shift=UP), FadeIn(desc6, shift=UP))
        self.next_slide()

        axes_3d = ThreeDAxes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            z_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            z_length=2.5,
            axis_config={"color": BLACK}
        )

        self.add(axes_3d)
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-45 * DEGREES,
            frame_center=axes_3d.get_center(),
        )
        self.next_slide()

        np.random.seed(24)
        num_points = 15
        x_vals = np.random.uniform(0, 5, num_points)
        z_vals = np.random.uniform(0, 5, num_points)
        y_vals = 1.0 + 0.2 * x_vals + 0.3 * z_vals + 0.1 * x_vals * z_vals
        y_vals += np.random.normal(0, 0.3, size=num_points)
        y_vals = np.clip(y_vals, 0, 5)

        points_3d = [Dot3D(axes_3d.c2p(x, z, y), color=BLUE, radius=0.07) for x, z, y in zip(x_vals, z_vals, y_vals)]
        self.play(LaggedStartMap(FadeIn, points_3d, shift=UP))
        self.next_slide()

        def surface_func(u, v):
            return 1.0 + 0.2 * u + 0.3 * v + 0.1 * u * v

        surface = Surface(
            lambda u, v: axes_3d.c2p(u, v, surface_func(u, v)),
            u_range=[0, 5],
            v_range=[0, 5],
            resolution=(15, 15),
            fill_color=RED,
            fill_opacity=0.5,
            stroke_color=BLACK,
            stroke_width=1
        )
        self.play(Create(surface), run_time=2)
        self.next_slide()

        self.move_camera(
            phi=70 * DEGREES,
            theta=120 * DEGREES,
            frame_center=axes_3d.get_center(),
            run_time=4
        )
        self.next_slide()

        formula6 = MathTex(
            r"\hat{y} = \theta_0 + \theta_1 x + \theta_2 z + \theta_3 x z",
            color=BLACK
        ).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(formula6)
        self.play(FadeIn(formula6, shift=UP))
        self.next_slide()

        self.play(FadeOut(VGroup(*points_3d, surface, axes_3d, formula6, title6, desc6)))

class EndSlide(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        q_text = Text("Questions & Discussion", font_size=64, color=ORANGE)
        self.play(FadeIn(q_text, shift=UP))
        self.next_slide()

        self.play(FadeOut(q_text))
        thank_text = Text("Thank you!", font_size=72, color=BLACK)
        self.play(FadeIn(thank_text, scale=1.2))
        self.next_slide()

        self.play(FadeOut(thank_text))