import numpy as np
import cv2
import torch

# DO NOT MODIFY THIS FILE
# in case of any bugs found add a ticket on github

class CannonEnv:
    cannon_speed_max = 25
    cannon_speed_min = 5
    cannon_speed_modifier = 10
    cannon_angle_max = torch.pi / 2
    cannon_angle_min = 0

    def __init__(self, device: str = "cpu"):
        self.canvas_width = torch.tensor(1200, device=device)
        self.canvas_height = torch.tensor(600, device=device)

        self.canon_width = torch.tensor(50, device=device)
        self.target_width = torch.randint(
            200, self.canvas_width - 100, (), device=device
        )
        self.gravity = torch.tensor(0.5, device=device)
        self.hit = False
        self.cannon_angle = torch.rand((), device=device) * CannonEnv.cannon_angle_max
        self.cannon_speed = torch.rand((), device=device) * (CannonEnv.cannon_speed_max - CannonEnv.cannon_speed_min) + CannonEnv.cannon_speed_min

    def _get_velocity(
        self, angle_diff: torch.Tensor, speed_diff: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method for calculating and updating of velocities
        """
        assert angle_diff.abs() <= 1, f"angle_diff cannot exceed 1 and is {angle_diff}"
        assert speed_diff.abs() <= 1, f"speed_diff cannot exceed 1 and is {speed_diff}"

        self.cannon_angle = torch.clamp(
            self.cannon_angle + angle_diff,
            CannonEnv.cannon_angle_min,
            CannonEnv.cannon_angle_max,
        )
        self.cannon_speed = torch.clamp(
            self.cannon_speed + speed_diff * CannonEnv.cannon_speed_modifier,
            CannonEnv.cannon_speed_min,
            CannonEnv.cannon_speed_max,
        )

        vx = self.cannon_speed * torch.cos(self.cannon_angle)
        vy = self.cannon_speed * torch.sin(self.cannon_angle)

        return vx, vy

    def fire(self, angle_diff: torch.Tensor, speed_diff: torch.Tensor):
        """
        Computes distance to target for specific shot. Results might differ slightly from show_fire, so take fire() as ground truth
        Returns a tensor of `[distance_to_target, self.cannon_angle, self.cannon_speed]`
        """
        vx, vy = self._get_velocity(angle_diff, speed_diff)
        time = 2 * vy / self.gravity
        distance_to_target = self.canon_width + vx * time - self.target_width
        return torch.stack([distance_to_target, self.cannon_angle, self.cannon_speed])

    def show_fire(
        self,
        angle_diff,
        speed_diff,
        debug_prints: bool = False,
        frame_duration: int = 30,
    ):
        """
        Simulates and displays shot trajectory. Results might differ slightly from fire(), so take fire() as ground truth
        Returns a tensor of `[distance_to_target, self.cannon_angle, self.cannon_speed]`
        """
        vx, vy = self._get_velocity(torch.tensor(angle_diff), torch.tensor(speed_diff))

        if debug_prints:
            print(f"Ball spawned with vx={vx.item()} and vy={vy.item()}")

        # Convert tensors to builtin python types
        canvas_height = self.canvas_height.type(torch.int).item()
        canvas_width = self.canvas_width.type(torch.int).item()
        cannon_width = self.canon_width.type(torch.int).item()
        target_width = self.target_width.type(torch.int).item()
        cannon_speed = self.cannon_speed.type(torch.int).item()

        ball_position = list((float(cannon_width), 0.0))

        time = 1

        # Create a blank canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Draw the base of the cannon
        cv2.circle(canvas, (cannon_width, canvas_height), 50, (0, 255, 0), -1)

        # Draw the target
        cv2.circle(canvas, (target_width, canvas_height), 10, (255, 0, 0), -1)

        # Main animation loop
        while True:
            frame = canvas.copy()

            # Calculate the end point of the rifle based on the cannon angle
            ball_position[0] = cannon_width + vx * time
            ball_position[1] = canvas_height - vy * time + 0.5 * self.gravity * time**2

            if debug_prints:
                print(
                    f"Ball is at: {ball_position[0].item(), ball_position[1].item()} at tick {time}"
                )

            # Draw the ball
            cv2.circle(
                frame, tuple(int(item) for item in ball_position), 2, (0, 0, 255), -1
            )

            # Draw the rifle
            rifle_length = 40 + cannon_speed * 3
            rifle_width = 5

            base_rectangle = torch.tensor(
                [
                    [0, rifle_width],
                    [rifle_length, rifle_width],
                    [rifle_length, -rifle_width],
                    [0, -rifle_width],
                ],
                dtype=torch.float32,
            )

            theta = -self.cannon_angle
            R = torch.tensor(
                [
                    [theta.cos(), -theta.sin()],
                    [theta.sin(), theta.cos()],
                ],
                dtype=torch.float32,
            )

            rotated = base_rectangle.matmul(R.T)
            cannon_base = torch.tensor(
                [cannon_width, canvas_height], dtype=torch.float32
            )
            shifted = rotated + cannon_base

            np_arr = shifted.cpu().numpy().astype(np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(frame, [np_arr], (0, 255, 0))

            cv2.imshow("Cannon Animation", frame)

            # Wait before next frame, or quit
            if cv2.waitKey(frame_duration) & 0xFF == ord("q"):
                break

            time += 1

            # if the ball goes below the screen, end the game
            if ball_position[1] >= canvas_height:
                distance_to_target = ball_position[0] - target_width
                if debug_prints:
                    print(f"Missed by: {distance_to_target} at tick {time}")

                if distance_to_target < 10:
                    self.hit = True

                return torch.stack(
                    [distance_to_target, self.cannon_angle, self.cannon_speed]
                )

    def cleanup(self):
        """
        Kills all cv2 windows. Use after showing everything needed with show_fire()
        """
        cv2.destroyAllWindows()

    def is_target_hit(self):
        """
        Will toggle to true if ball lands sufficiently close to the target. If that
        Happens, CannonEnv instance should be discarded and new one created
        """
        return self.hit


if __name__ == "__main__":
    torch.manual_seed(1)
    cannon_env = CannonEnv()
    print(cannon_env.show_fire(0, 0))
    print(cannon_env.show_fire(0, 0.5))
    print(cannon_env.show_fire(0.5, 0))
    cannon_env.cleanup()
