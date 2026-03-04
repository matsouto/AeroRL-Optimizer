import unittest
import numpy as np
from unittest.mock import Mock, patch
from gymnasium import spaces

from src.enviroments.static_opt_env import StaticOptEnv


class TestStaticOptEnvBase(unittest.TestCase):
    """Base class for StaticOptEnv tests with common setup."""

    @staticmethod
    def create_mock_scaler():
        """Create a mock scaler."""
        scaler = Mock()

        def inverse_transform(w_norm, p_norm):
            return w_norm * 0.5, p_norm * 0.5

        scaler.inverse_transform = inverse_transform
        return scaler

    @staticmethod
    def create_mock_onnx_session():
        """Create a mock ONNX session."""
        session = Mock()
        session.get_inputs.return_value = [Mock(name="input_0")]

        def run_mock(_, input_dict):
            batch_size = list(input_dict.values())[0].shape[0]
            w_norm = np.random.randn(batch_size, 8).astype(np.float32)
            p_norm = np.random.randn(batch_size, 8).astype(np.float32)
            return [w_norm, p_norm]

        session.run = run_mock
        return session

    def setup_patches(self):
        """Setup all patches for mocking."""
        self.mock_scaler = self.create_mock_scaler()
        self.mock_onnx_session = self.create_mock_onnx_session()

        # Create airfoil mock that returns the configured instance when called
        airfoil_mock_class = Mock()
        airfoil_instance = self._create_airfoil_mock()
        airfoil_mock_class.return_value = airfoil_instance

        self.patches = [
            patch("joblib.load", return_value=self.mock_scaler),
            patch("onnxruntime.InferenceSession", return_value=self.mock_onnx_session),
            patch("src.env.static_opt_env.Airfoil", airfoil_mock_class),
            patch(
                "src.env.static_opt_env.cst_to_coords",
                side_effect=self._cst_coords_mock,
            ),
            patch("src.env.static_opt_env.plt"),
        ]

        for p in self.patches:
            p.start()

    def teardown_patches(self):
        """Stop all patches."""
        for p in self.patches:
            p.stop()

    @staticmethod
    def _create_airfoil_mock():
        """Create airfoil mock."""
        airfoil_instance = Mock()

        def get_aero_mock(alpha, Re, mach, model_size):
            n_alphas = len(alpha)
            return {
                "CL": np.random.uniform(0.2, 1.5, n_alphas),
                "CD": np.random.uniform(0.01, 0.05, n_alphas),
                "analysis_confidence": np.random.uniform(0.5, 1.0, n_alphas),
            }

        airfoil_instance.get_aero_from_neuralfoil = get_aero_mock
        return airfoil_instance

    @staticmethod
    def _cst_coords_mock(w, p, n_points):
        """Mock CST coordinates."""
        x = np.linspace(0, 1, n_points)
        y = np.sin(x * np.pi) * 0.1
        return x, y


class TestStaticOptEnvInitialization(TestStaticOptEnvBase):
    """Test environment initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
            latent_dim=16,
            action_range=0.1,
            latent_range=3.0,
            max_episode_steps=50,
            n_alphas=40,
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_init_creates_env(self):
        """Test that environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.latent_dim, 16)
        self.assertEqual(self.env.action_range, 0.1)
        self.assertEqual(self.env.latent_range, 3.0)
        self.assertEqual(self.env.max_episode_steps, 50)
        self.assertEqual(self.env.n_alphas, 40)

    def test_action_space(self):
        """Test action space is correctly defined."""
        self.assertIsInstance(self.env.action_space, spaces.Box)
        self.assertEqual(self.env.action_space.shape, (16,))
        self.assertTrue(np.allclose(self.env.action_space.low, -0.1))
        self.assertTrue(np.allclose(self.env.action_space.high, 0.1))

    def test_observation_space(self):
        """Test observation space is correctly defined."""
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (16,))
        self.assertTrue(np.allclose(self.env.observation_space.low, -3.0))
        self.assertTrue(np.allclose(self.env.observation_space.high, 3.0))

    def test_initial_state(self):
        """Test initial state values."""
        self.assertEqual(self.env._current_step, 0)
        self.assertEqual(self.env._current_z.shape, (16,))
        self.assertEqual(self.env._current_cl_sweep.shape, (40,))
        self.assertEqual(self.env._current_cd_sweep.shape, (40,))
        self.assertIsInstance(self.env._current_efficiency, float)


class TestReset(TestStaticOptEnvBase):
    """Test reset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_reset_returns_observation_and_info(self):
        """Test reset returns properly formatted outputs."""
        observation, info = self.env.reset(seed=42)

        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (16,))
        self.assertIsInstance(info, dict)

    def test_reset_resets_step_counter(self):
        """Test that reset resets the step counter."""
        self.env._current_step = 10
        self.env.reset()
        self.assertEqual(self.env._current_step, 0)

    def test_reset_initializes_latent_vector(self):
        """Test that reset initializes a new latent vector."""
        self.env.reset(seed=None)
        self.assertEqual(self.env._current_z.shape, (16,))
        self.assertEqual(self.env._current_z.dtype, np.float32)

    def test_reset_with_seed_reproducibility(self):
        """Test reset with seed produces reproducible results."""
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=42)
        self.assertTrue(np.allclose(obs1, obs2))

    def test_reset_info_contains_expected_keys(self):
        """Test info dict contains expected keys."""
        _, info = self.env.reset()
        self.assertIn("cl_sweep", info)
        self.assertIn("cd_sweep", info)
        self.assertIn("efficiency", info)


class TestStep(TestStaticOptEnvBase):
    """Test step functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )
        self.env.reset()

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_step_returns_correct_types(self):
        """Test step returns correct types."""
        action = self.env.action_space.sample()

        observation, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIsInstance(info, dict)

    def test_step_observation_shape(self):
        """Test step returns observation with correct shape."""
        action = self.env.action_space.sample()
        observation, _, _, _, _ = self.env.step(action)

        self.assertEqual(observation.shape, (16,))

    def test_step_increments_counter(self):
        """Test step increments the step counter."""
        initial_step = self.env._current_step
        self.env.step(self.env.action_space.sample())
        self.assertEqual(self.env._current_step, initial_step + 1)

    def test_step_truncation_at_max_steps(self):
        """Test episode truncates at max_episode_steps."""
        for _ in range(self.env.max_episode_steps - 1):
            _, _, terminated, truncated, _ = self.env.step(
                self.env.action_space.sample()
            )
            self.assertFalse(truncated)

        _, _, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
        self.assertTrue(truncated)

    def test_step_action_updates_latent_vector(self):
        """Test step updates latent vector with action."""
        initial_z = self.env._current_z.copy()

        action = np.ones(16, dtype=np.float32) * 0.05
        self.env.step(action)

        self.assertFalse(np.allclose(self.env._current_z, initial_z))

    def test_step_clips_latent_vector(self):
        """Test latent vector is clipped within bounds."""
        large_action = np.ones(16, dtype=np.float32) * 10.0

        for _ in range(100):
            self.env.step(large_action)

        self.assertTrue(np.all(self.env._current_z >= -self.env.latent_range))
        self.assertTrue(np.all(self.env._current_z <= self.env.latent_range))

    def test_step_reward_is_scalar(self):
        """Test step returns scalar reward."""
        _, reward, _, _, _ = self.env.step(self.env.action_space.sample())

        self.assertIsInstance(reward, (float, np.floating))
        self.assertNotIsInstance(reward, np.ndarray)

    def test_step_negative_reward_on_invalid_profile(self):
        """Test negative reward when profile is invalid."""
        with patch("src.env.static_opt_env.Airfoil") as mock_airfoil:
            mock_airfoil.side_effect = Exception("Invalid airfoil")

            _, reward, _, _, _ = self.env.step(self.env.action_space.sample())

            self.assertEqual(reward, -50.0)

    def test_step_updates_aero_coefficients(self):
        """Test step updates aerodynamic coefficients."""
        self.env._current_cl_sweep = np.zeros(self.env.n_alphas)
        self.env._current_cd_sweep = np.zeros(self.env.n_alphas)

        self.env.step(self.env.action_space.sample())

        # After step, CL sweep should have been updated (mocked data is always valid with high confidence)
        self.assertTrue(np.any(self.env._current_cl_sweep != 0))

    def test_step_calculates_efficiency(self):
        """Test step calculates efficiency correctly."""
        self.env.step(self.env.action_space.sample())

        self.assertIsInstance(self.env._current_efficiency, (float, np.floating))
        self.assertTrue(
            np.isfinite(self.env._current_efficiency)
            or self.env._current_efficiency < -10.0
        )

    def test_step_info_dict(self):
        """Test step info contains aerodynamic data."""
        _, _, _, _, info = self.env.step(self.env.action_space.sample())

        self.assertIn("cl_sweep", info)
        self.assertIn("cd_sweep", info)
        self.assertIn("efficiency", info)
        self.assertEqual(len(info["cl_sweep"]), self.env.n_alphas)
        self.assertEqual(len(info["cd_sweep"]), self.env.n_alphas)


class TestGetObs(TestStaticOptEnvBase):
    """Test observation retrieval."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )
        self.env.reset()

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_get_obs_returns_latent_vector(self):
        """Test _get_obs returns the latent vector."""
        self.env._current_z = np.arange(16, dtype=np.float32)

        obs = self.env._get_obs()
        self.assertTrue(np.allclose(obs, self.env._current_z))


class TestGetCoords(TestStaticOptEnvBase):
    """Test coordinate generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )
        self.env.reset()

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_get_coords_returns_array(self):
        """Test _get_coords returns coordinate array."""
        coords = self.env._get_coords()

        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(coords.shape[1], 2)

    def test_get_coords_shape(self):
        """Test coordinates have expected shape."""
        coords = self.env._get_coords()

        self.assertEqual(len(coords.shape), 2)
        self.assertEqual(coords.shape[1], 2)


class TestRender(TestStaticOptEnvBase):
    """Test render functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )
        self.env.reset()

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_render_does_not_crash(self):
        """Test render method runs without error."""
        self.env.step(self.env.action_space.sample())
        self.env.render()


class TestClose(TestStaticOptEnvBase):
    """Test cleanup functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )

    def tearDown(self):
        """Clean up after tests."""
        self.teardown_patches()

    def test_close_completes(self):
        """Test close method completes without error."""
        self.env.close()


class TestEpisodeLoop(TestStaticOptEnvBase):
    """Test complete episode execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_full_episode(self):
        """Test running a complete episode."""
        observation, info = self.env.reset(seed=42)

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 60:
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated
            total_reward += reward
            steps += 1

            self.assertEqual(observation.shape, (16,))
            self.assertIsInstance(reward, (float, np.floating))

        self.assertGreater(steps, 0)
        self.assertGreater(self.env._current_step, 0)

    def test_multiple_episodes(self):
        """Test multiple episodes in sequence."""
        for episode in range(3):
            observation, info = self.env.reset(seed=episode)

            for step in range(5):
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)

            if episode > 0:
                self.assertEqual(self.env._current_step, 5)


class TestCustomParameters(TestStaticOptEnvBase):
    """Test environment with custom parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()

    def tearDown(self):
        """Clean up after tests."""
        self.teardown_patches()

    def test_custom_latent_dim(self):
        """Test environment with custom latent dimension."""
        env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
            latent_dim=32,
        )
        self.assertEqual(env.action_space.shape, (32,))
        env.close()

    def test_custom_episode_length(self):
        """Test environment with custom max episode steps."""
        env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
            max_episode_steps=100,
        )
        self.assertEqual(env.max_episode_steps, 100)

        env.reset()
        for _ in range(99):
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            self.assertFalse(truncated)

        _, _, _, truncated, _ = env.step(env.action_space.sample())
        self.assertTrue(truncated)
        env.close()


class TestActionSpaceSampling(TestStaticOptEnvBase):
    """Test action space sampling."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_action_space_sample_validity(self):
        """Test sampled actions are within bounds."""
        for _ in range(10):
            action = self.env.action_space.sample()
            self.assertTrue(self.env.action_space.contains(action))
            self.assertTrue(np.all(action >= -self.env.action_range))
            self.assertTrue(np.all(action <= self.env.action_range))


class TestObservationSpaceBounds(TestStaticOptEnvBase):
    """Test observation space bounds."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_observation_in_space(self):
        """Test observation is within observation space."""
        obs, _ = self.env.reset()

        self.assertEqual(obs.shape, (16,))


class TestDataTypes(TestStaticOptEnvBase):
    """Test data types are correct."""

    def setUp(self):
        """Set up test fixtures."""
        self.setup_patches()
        self.env = StaticOptEnv(
            scaler_path="dummy_scaler.pkl",
            decoder_path="dummy_decoder.onnx",
        )
        self.env.reset()

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        self.teardown_patches()

    def test_observation_dtype(self):
        """Test observation has correct dtype."""
        obs, _ = self.env.reset()
        self.assertEqual(obs.dtype, np.float32)

    def test_action_space_dtype(self):
        """Test action space has correct dtype."""
        self.assertEqual(self.env.action_space.dtype, np.float32)

    def test_observation_space_dtype(self):
        """Test observation space has correct dtype."""
        self.assertEqual(self.env.observation_space.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
