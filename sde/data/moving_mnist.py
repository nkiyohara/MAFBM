import numpy as np
import math
from torchvision import datasets, transforms


# class MovingMNIST(object):

#     """Data Handler that creates Bouncing MNIST dataset on the fly."""

#     def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
#         path = data_root
#         self.seq_len = seq_len
#         self.num_digits = num_digits
#         self.image_size = image_size
#         self.step_length = 0.1
#         self.digit_size = 32
#         self.deterministic = deterministic
#         self.seed_is_set = False # multi threaded loading
#         self.channels = 1

#         self.data = datasets.MNIST(
#             path,
#             train=train,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.Resize(self.digit_size),
#                  transforms.ToTensor()]))

#         self.N = len(self.data)

#     def set_seed(self, seed):
#         if not self.seed_is_set:
#             self.seed_is_set = True
#             np.random.seed(seed)

#     def __len__(self):
#         return self.N

#     def __getitem__(self, index):
#         self.set_seed(index)
#         image_size = self.image_size
#         digit_size = self.digit_size
#         x = np.zeros((self.seq_len,
#                       image_size,
#                       image_size,
#                       self.channels),
#                     dtype=np.float32)
#         for n in range(self.num_digits):
#             idx = np.random.randint(self.N)
#             # idx = 0
#             digit, _ = self.data[idx]

#             sx = np.random.randint(image_size-digit_size)
#             sy = np.random.randint(image_size-digit_size)
#             dx = np.random.randint(-4, 5)
#             dy = np.random.randint(-4, 5)
#             for t in range(self.seq_len):
#                 if sy < 0:
#                     sy = 0
#                     if self.deterministic:
#                         dy = -dy
#                     else:
#                         dy = np.random.randint(1, 5)
#                         dx = np.random.randint(-4, 5)
#                 elif sy >= image_size-32:
#                     sy = image_size-32-1
#                     if self.deterministic:
#                         dy = -dy
#                     else:
#                         dy = np.random.randint(-4, 0)
#                         dx = np.random.randint(-4, 5)

#                 if sx < 0:
#                     sx = 0
#                     if self.deterministic:
#                         dx = -dx
#                     else:
#                         dx = np.random.randint(1, 5)
#                         dy = np.random.randint(-4, 5)
#                 elif sx >= image_size-32:
#                     sx = image_size-32-1
#                     if self.deterministic:
#                         dx = -dx
#                     else:
#                         dx = np.random.randint(-4, 0)
#                         dy = np.random.randint(-4, 5)

#                 x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
#                 sy += dy
#                 sx += dx

#         x[x>1] = 1.
#         return x


class MovingMNIST(object):
    """Data Handler that creates Bouncing MNIST dataset with physically accurate reflections.

    This implementation maintains the same interface as the original MovingMNIST
    but uses improved physics-based collision detection and reflection with
    Gaussian noise to simulate realistic bouncing behavior.
    """

    def __init__(
        self,
        train,
        data_root,
        seq_len=20,
        num_digits=2,
        image_size=64,
        deterministic=True,
        angle_noise_sigma=0.1,
    ):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.angle_noise_sigma = angle_noise_sigma
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size), transforms.ToTensor()]
            ),
        )
        self.N = len(self.data)

    def set_seed(self, seed):
        # This method exists for compatibility but we'll use RandomState
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def _detect_collision(self, sx, sy, dx, dy, image_size, digit_size):
        """Detect wall collisions within a time step and calculate the exact collision time.

        Returns:
            collision_time: Time of collision (0~1, where 1 is the next frame). None if no collision.
            wall_type: Type of wall hit ("left", "right", "top", "bottom")
        """
        collision_times = []
        wall_types = []

        # Check left wall collision
        if dx < 0:
            time_to_left = -sx / dx
            if 0 <= time_to_left <= 1:
                collision_times.append(time_to_left)
                wall_types.append("left")

        # Check right wall collision
        if dx > 0:
            time_to_right = (image_size - digit_size - sx) / dx
            if 0 <= time_to_right <= 1:
                collision_times.append(time_to_right)
                wall_types.append("right")

        # Check top wall collision
        if dy < 0:
            time_to_top = -sy / dy
            if 0 <= time_to_top <= 1:
                collision_times.append(time_to_top)
                wall_types.append("top")

        # Check bottom wall collision
        if dy > 0:
            time_to_bottom = (image_size - digit_size - sy) / dy
            if 0 <= time_to_bottom <= 1:
                collision_times.append(time_to_bottom)
                wall_types.append("bottom")

        # No collision
        if not collision_times:
            return None, None

        # Return earliest collision
        earliest_idx = np.argmin(collision_times)
        return collision_times[earliest_idx], wall_types[earliest_idx]

    def _apply_reflection_with_noise(self, dx, dy, wall_type, rng):
        """Apply physics-based reflection with Gaussian noise to the angle.

        Args:
            dx, dy: Current velocity components
            wall_type: Type of wall hit ("left", "right", "top", "bottom")
            rng: Random number generator

        Returns:
            new_dx, new_dy: New velocity components after reflection and noise
        """
        # Current speed and angle
        speed = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        # Calculate reflection angle based on wall
        if wall_type in ["left", "right"]:
            # Reflect across y-axis for left/right walls
            reflected_angle = math.pi - angle
        else:  # top or bottom
            # Reflect across x-axis for top/bottom walls
            reflected_angle = -angle

        # Add Gaussian noise to reflected angle
        if not self.deterministic:
            noise = rng.normal(0, self.angle_noise_sigma)
            noisy_angle = reflected_angle + noise
        else:
            noisy_angle = reflected_angle

        # Convert back to velocity components
        new_dx = speed * math.cos(noisy_angle)
        new_dy = speed * math.sin(noisy_angle)

        # Ensure digit moves away from the wall
        if wall_type == "left" and new_dx < 0:
            new_dx = abs(new_dx)
        elif wall_type == "right" and new_dx > 0:
            new_dx = -abs(new_dx)
        elif wall_type == "top" and new_dy < 0:
            new_dy = abs(new_dy)
        elif wall_type == "bottom" and new_dy > 0:
            new_dy = -abs(new_dy)

        return new_dx, new_dy

    def __getitem__(self, index):
        # Create a random state using the index as seed for reproducibility
        rng = np.random.RandomState(index)
        self.set_seed(index)  # For compatibility

        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros(
            (self.seq_len, image_size, image_size, self.channels), dtype=np.float32
        )

        for n in range(self.num_digits):
            idx = rng.randint(self.N)
            digit, _ = self.data[idx]

            # Initialize position and velocity
            sx = float(rng.randint(image_size - digit_size))
            sy = float(rng.randint(image_size - digit_size))

            # Use a speed and angle for more controlled initialization
            speed = rng.uniform(1.0, 5.0)
            angle = rng.uniform(0, 2 * math.pi)
            dx = speed * math.cos(angle)
            dy = speed * math.sin(angle)

            for t in range(self.seq_len):
                # Add digit to current frame
                x_int, y_int = int(sx), int(sy)
                x_int = max(0, min(image_size - digit_size, x_int))
                y_int = max(0, min(image_size - digit_size, y_int))
                x[t, y_int : y_int + digit_size, x_int : x_int + digit_size, 0] += (
                    digit.numpy().squeeze()
                )

                # Detect collision within time step
                collision_time, wall_type = self._detect_collision(
                    sx, sy, dx, dy, image_size, digit_size
                )

                if collision_time is not None:
                    # Move to collision point
                    sx_collision = sx + collision_time * dx
                    sy_collision = sy + collision_time * dy

                    # Calculate reflection angle
                    new_dx, new_dy = self._apply_reflection_with_noise(
                        dx, dy, wall_type, rng
                    )

                    # Move in new direction for remaining time
                    remaining_time = 1.0 - collision_time
                    sx = sx_collision + remaining_time * new_dx
                    sy = sy_collision + remaining_time * new_dy

                    # Update velocity
                    dx, dy = new_dx, new_dy
                else:
                    # Simple movement if no collision
                    sx += dx
                    sy += dy

                # Safety boundary check (for numerical errors)
                if sx < 0:
                    sx = 0
                    dx = -dx if self.deterministic else abs(dx)
                elif sx > image_size - digit_size:
                    sx = image_size - digit_size
                    dx = -dx if self.deterministic else -abs(dx)

                if sy < 0:
                    sy = 0
                    dy = -dy if self.deterministic else abs(dy)
                elif sy > image_size - digit_size:
                    sy = image_size - digit_size
                    dy = -dy if self.deterministic else -abs(dy)

        x[x > 1] = 1.0
        return x

    def get_trajectory_with_different_noise(
        self,
        index: int,
        noise_seed: int,
        divergence_step: int = 0,
        swap_digits: bool = False,
    ):
        """Generate a trajectory with controlled noise divergence.

        This method creates a trajectory that follows the same path as the one from __getitem__(index)
        up to the divergence_step, after which it applies different reflection noise.

        Args:
            index: The index determining the base trajectory
            noise_seed: Seed for the random number generator used for reflection noise
            divergence_step: The step at which the trajectory starts to diverge (0-indexed).
                             Default is 0, meaning the entire trajectory will have different noise.

        Returns:
            Numpy array with shape (seq_len, image_size, image_size, channels)
        """
        # Use index to determine the base trajectory
        base_rng = np.random.RandomState(index)
        # Use separate RNG for reflection noise after divergence
        noise_rng = np.random.RandomState(noise_seed)

        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros(
            (self.seq_len, image_size, image_size, self.channels), dtype=np.float32
        )
        indices = base_rng.randint(self.N, size=self.num_digits)
        if swap_digits:
            indices = indices[::-1]

        for n in range(self.num_digits):
            idx = indices[n]
            digit, _ = self.data[idx]

            # Initialize position and velocity
            sx = float(base_rng.randint(image_size - digit_size))
            sy = float(base_rng.randint(image_size - digit_size))
            speed = base_rng.uniform(1.0, 5.0)
            angle = base_rng.uniform(0, 2 * math.pi)
            dx = speed * math.cos(angle)
            dy = speed * math.sin(angle)

            for t in range(self.seq_len):
                # Add digit to current frame
                x_int, y_int = int(sx), int(sy)
                x_int = max(0, min(image_size - digit_size, x_int))
                y_int = max(0, min(image_size - digit_size, y_int))
                x[t, y_int : y_int + digit_size, x_int : x_int + digit_size, 0] += (
                    digit.numpy().squeeze()
                )

                # Detect collision within time step
                collision_time, wall_type = self._detect_collision(
                    sx, sy, dx, dy, image_size, digit_size
                )

                if collision_time is not None:
                    # Move to collision point
                    sx_collision = sx + collision_time * dx
                    sy_collision = sy + collision_time * dy

                    # Choose which RNG to use based on current step
                    # Before divergence_step: use base_rng to match original trajectory
                    # After divergence_step: use noise_rng for different behavior
                    current_rng = base_rng if t < divergence_step else noise_rng

                    # Calculate reflection angle
                    new_dx, new_dy = self._apply_reflection_with_noise(
                        dx, dy, wall_type, current_rng
                    )

                    # Move in new direction for remaining time
                    remaining_time = 1.0 - collision_time
                    sx = sx_collision + remaining_time * new_dx
                    sy = sy_collision + remaining_time * new_dy

                    # Update velocity
                    dx, dy = new_dx, new_dy
                else:
                    # Simple movement if no collision
                    sx += dx
                    sy += dy

                # Safety boundary check (for numerical errors)
                if sx < 0:
                    sx = 0
                    dx = -dx if self.deterministic else abs(dx)
                elif sx > image_size - digit_size:
                    sx = image_size - digit_size
                    dx = -dx if self.deterministic else -abs(dx)

                if sy < 0:
                    sy = 0
                    dy = -dy if self.deterministic else abs(dy)
                elif sy > image_size - digit_size:
                    sy = image_size - digit_size
                    dy = -dy if self.deterministic else -abs(dy)

        # Clip values
        x[x > 1] = 1.0
        return x
