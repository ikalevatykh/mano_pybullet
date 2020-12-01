"""Animated GIF recorder."""

try:
    import imageio
except ModuleNotFoundError as module_not_found:
    raise RuntimeError(
        'Package not found: imageio. Use `pip install imageio` to install.') from module_not_found

import gym

__all__ = ('GIFRecorder')


class GIFRecorder(gym.Wrapper):
    """Animated GIF recorder."""

    def __init__(self, env, filename_template, use_seed=False):
        """Constructor for a AnimRecorder.

        Arguments:
            env {gym.Env} -- environment to wrap
            filename_template {str} -- GIF filename template (like 'record_{:04d}.gif').

        Keyword Arguments:
            use_seed {bool} -- use seed as file id (substitute in the template) (default: {False}).
        """
        super().__init__(env)
        self.env.unwrapped.show_window(False)
        self._filename_template = filename_template
        self._use_seed = use_seed

        input_fps = self.env.unwrapped.metadata.get('video.frames_per_second', 30.0)
        output_fps = self.env.unwrapped.metadata.get('video.output_frames_per_second', 30.0)
        self._frames_per_second = output_fps
        self._steps_per_frame = max(1, input_fps // output_fps)

        self._reset_counter = 0
        self._step_counter = 0
        self._writer = None

    def reset(self, **kwargs):
        """Resets the environment to an initial state.

        Open a GIF file writer.
        """
        if self._writer is not None:
            self._writer.close()

        uid = self._seed if self._use_seed else self._reset_counter
        filename = self._filename_template.format(uid)
        self._writer = imageio.get_writer(filename, mode='I', fps=self._frames_per_second)

        self._reset_counter += 1
        self._step_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Render an image and write it to the file.
        """
        if self._step_counter % self._steps_per_frame == 0:
            image_rgb = self.env.render('rgb_array')
            self._writer.append_data(image_rgb)
        self._step_counter += 1
        return self.env.step(action)

    def close(self):
        """Close the environment.

        Flush recorded data to file.
        """
        if self._writer is not None:
            self._writer.close()
        return self.env.close()
