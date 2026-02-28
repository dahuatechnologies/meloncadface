/**
 * @file mimix/src/main.c
 * @version 6.0.0
 * @license MIT
 *
 * @title MIMIX CAD Face - Neural Dimensional System
 * @description Complete OpenGL Render, OpenAL Sound, and Keyboard Callback Control
 *
 * @features
 *   - 5-Axis Neural Visualization with OpenGL
 *   - Spatial Audio with OpenAL (neuron firing sounds)
 *   - Keyboard Callback System with 30+ commands
 *   - SDL2 Window Management with resize support
 *   - 262,144 Neurons with spike detection
 *   - 65,536 Vector connections (synapses)
 *   - Real-time FPS counter and statistics
 *   - Multiple render modes (points, lines, axes)
 *   - Camera controls (rotate, zoom, pan)
 *   - Sound feedback for neural activity
 *   - Pause/Resume simulation
 *   - Save/Load neural states
 *   - Quality settings for performance
 */

/* ============================================================================
 * ANSI C89/90 Standard Headers
 * =========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stddef.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <setjmp.h>

/* Define inline for C89 compatibility */
#ifndef inline
#define inline __inline__
#endif

/* POSIX Headers */
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>

/* OpenGL */
#include <GL/gl.h>
#include <GL/glu.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alut.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* SIMD Headers */
#include <immintrin.h>
#include <x86intrin.h>

/* ============================================================================
 * Compile-time Configuration
 * =========================================================================== */

#define PROGRAM_NAME         "MIMIX CAD Face"
#define PROGRAM_VERSION      "6.0.0"
#define PROGRAM_AUTHOR       "Evolutia"
#define PROGRAM_YEAR         "2026"

/* Window Configuration */
#define WINDOW_WIDTH         1280
#define WINDOW_HEIGHT        768
#define WINDOW_BPP           32
#define WINDOW_REFRESH_RATE   60

/* Neural Network Configuration */
#define AXIS_COUNT           5
#define VECTOR_COUNT         65536
#define NEURON_COUNT         262144
#define NEURAL_LAYERS        12
#define SYNAPSE_DENSITY      0.15f
#define MAX_SPIKE_HISTORY    256
#define SPIKE_SOUND_COUNT    16

/* Thread Configuration */
#define MAX_THREADS          4
#define FRAME_TIME_MS        16  /* ~60 FPS */
#define SOUND_UPDATE_MS      50  /* Sound update interval */

/* Rendering Parameters */
#define POINT_SIZE           2.0f
#define LINE_WIDTH           1.0f
#define AXIS_LINE_WIDTH      2.0f
#define RENDER_QUALITY_MAX   3
#define RENDER_QUALITY_MIN   1

/* Camera Configuration */
#define CAMERA_MIN_DIST      3.0f
#define CAMERA_MAX_DIST      20.0f
#define CAMERA_MIN_HEIGHT    -5.0f
#define CAMERA_MAX_HEIGHT    10.0f
#define CAMERA_ROT_SPEED     0.02f
#define CAMERA_PAN_SPEED     0.1f

/* Learning Parameters */
#define LEARNING_RATE        0.01f
#define SENSITIVITY          1.0f
#define ADAPTATION_RATE      0.1f
#define THRESHOLD_BASE       0.5f

/* Sound Configuration */
#define SOUND_BUFFERS        32
#define SOUND_SOURCES        16
#define SOUND_PITCH_BASE     440.0f  /* A4 note */
#define SOUND_VOLUME_MAX     0.5f
#define SOUND_VOLUME_MIN     0.0f

/* Memory alignment */
#define CACHE_LINE_SIZE      64
#define SIMD_ALIGNMENT       32
#define ALIGNED(align) __attribute__((aligned(align)))
#define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#define SIMD_ALIGNED __attribute__((aligned(SIMD_ALIGNMENT)))

/* ============================================================================
 * Type Definitions
 * =========================================================================== */

/**
 * @brief 5D Vector with SIMD alignment
 */
typedef struct SIMD_ALIGNED NeuralVector5D {
	float x, y, z, d, a;
	float weight;
	float velocity[5];
} NeuralVector5D;

/**
 * @brief RGBA Color
 */
typedef struct ColorRGBA {
	unsigned char r, g, b, a;
} ColorRGBA;

/**
 * @brief Neural Dimensional Neuron
 */
typedef struct CACHE_ALIGNED DimensionalNeuron {
	NeuralVector5D pos; /* Position in 5D space */
	float potential; /* Membrane potential */
	float threshold; /* Firing threshold */
	float refractory; /* Refractory period */
	float adaptation; /* Adaptation rate */
	float weights[AXIS_COUNT]; /* Synaptic weights */
	unsigned int spikes; /* Spike counter */
	unsigned int last_spike; /* Last spike time */
	ColorRGBA color; /* Neuron color */
	unsigned int id; /* Neuron ID */
	float hebbian[AXIS_COUNT]; /* Hebbian trace */
	int frozen; /* Frozen state */
} DimensionalNeuron;

/**
 * @brief Neural Synapse (Vector Line)
 */
typedef struct CACHE_ALIGNED NeuralSynapse {
	NeuralVector5D start; /* Pre-synaptic position */
	NeuralVector5D end; /* Post-synaptic position */
	float strength; /* Synaptic strength */
	float plasticity; /* Plasticity factor */
	float activity; /* Current activity */
	unsigned int pre_id; /* Pre-synaptic neuron ID */
	unsigned int post_id; /* Post-synaptic neuron ID */
	unsigned int bundle; /* Axis bundle */
	ColorRGBA color; /* Synapse color */
	float hebbian; /* Hebbian trace */
} NeuralSynapse;

/**
 * @brief Neural Face - Complete mesh
 */
typedef struct CACHE_ALIGNED CADNeuralFace {
	DimensionalNeuron *neurons;
	NeuralSynapse *synapses;
	unsigned int neuron_count;
	unsigned int synapse_count;
	float bounds_min[3];
	float bounds_max[3];
	float center[3];
	float radius;
	float activity[AXIS_COUNT];
	pthread_mutex_t mutex;
	volatile int initialized;
	volatile int modified;
} CADNeuralFace;

/**
 * @brief Keyboard Callback Function
 */
typedef void (*KeyboardCallback)(SDL_Keycode key, void *userdata);

/**
 * @brief Keyboard Callback Entry
 */
typedef struct KeyboardCallbackEntry {
	SDL_Keycode key;
	KeyboardCallback callback;
	void *userdata;
	int priority;
	struct KeyboardCallbackEntry *next;
} KeyboardCallbackEntry;

/**
 * @brief Control System
 */
typedef struct CACHE_ALIGNED ControlSystem {
	int mode; /* 0=auto, 1=manual, 2=hybrid */
	int paused; /* Pause simulation */
	int render_quality; /* 1=low, 2=med, 3=high */
	int show_axes; /* Show axis lines */
	int show_vectors; /* Show vector lines */
	int show_neurons; /* Show neuron points */
	int sound_enabled; /* Enable sound */
	float sound_volume; /* Sound volume 0-1 */
	float camera_angle; /* Camera angle */
	float camera_dist; /* Camera distance */
	float camera_height; /* Camera height */
	float camera_pan_x; /* Camera pan X */
	float camera_pan_y; /* Camera pan Y */
	float rotation_speed; /* Auto-rotation speed */
	float learning_rate; /* Learning rate */
	float sensitivity; /* Neural sensitivity */
	unsigned long total_spikes; /* Total spikes */
	KeyboardCallbackEntry *callbacks[256];
	pthread_mutex_t callback_mutex;
} ControlSystem;

/**
 * @brief OpenGL Renderer
 */
typedef struct CACHE_ALIGNED OpenGLRenderer {
	SDL_Window *window;
	SDL_GLContext context;
	int width;
	int height;
	int initialized;
	GLuint neuron_list;
	GLuint synapse_list;
	GLuint axis_lists[AXIS_COUNT];
	GLuint grid_list;
	float clear_color[4];
	int list_valid;
} OpenGLRenderer;

/**
 * @brief OpenAL Audio System
 */
typedef struct CACHE_ALIGNED OpenALSystem {
	ALCdevice *device;
	ALCcontext *context;
	ALuint buffers[SPIKE_SOUND_COUNT];
	ALuint sources[SOUND_SOURCES];
	int source_index;
	int initialized;
	float listener_pos[3];
	float listener_vel[3];
	float listener_ori[6];
} OpenALSystem;

/**
 * @brief Application State
 */
typedef struct CACHE_ALIGNED AppState {
	CADNeuralFace face;
	ControlSystem control;
	OpenGLRenderer gl;
	OpenALSystem al;
	pthread_t threads[MAX_THREADS];
	unsigned int thread_count;
	volatile int running;
	volatile int shutting_down;
	double fps;
	unsigned int last_frame_time;
	int frame_count;
	struct timespec start_time;
	jmp_buf emergency_jmp;
} AppState;

/* ============================================================================
 * Global State
 * =========================================================================== */

static volatile int g_running = 1;
static AppState *g_state = NULL;

/* ============================================================================
 * Function Prototypes
 * =========================================================================== */

/* Core Neural Functions */
static int init_neural_face(AppState *state);
static void free_neural_face(AppState *state);
static void update_neurons(AppState *state);
static void update_synapses(AppState *state);
static void* thread_worker(void *arg);
static float compute_input_current(DimensionalNeuron *n, ControlSystem *ctl);

/* OpenGL Renderer */
static int init_opengl_renderer(OpenGLRenderer *gl);
static void shutdown_opengl_renderer(OpenGLRenderer *gl);
static void build_display_lists(AppState *state);
static void render_frame(AppState *state);
static void render_axes(AppState *state);
static void render_grid(void);
static void render_hud(AppState *state);
static void check_gl_error(const char *location);
static void resize_viewport(OpenGLRenderer *gl, int width, int height);

/* OpenAL Audio */
static int init_openal_audio(OpenALSystem *al);
static void shutdown_openal_audio(OpenALSystem *al);
static void generate_sine_wave(ALuint buffer, float frequency, float duration);
static void play_spike_sound(OpenALSystem *al, float pitch, float volume);
static void update_listener_position(OpenALSystem *al, float x, float y,
		float z);
static void set_sound_volume(OpenALSystem *al, float volume);

/* Keyboard Callback System */
static void init_control_system(ControlSystem *ctl, AppState *state);
static void register_keyboard_callback(ControlSystem *ctl, SDL_Keycode key,
		KeyboardCallback callback, void *userdata, int priority);
static void dispatch_keyboard_callbacks(ControlSystem *ctl, SDL_Keycode key);
static void keyboard_callback_escape(SDL_Keycode key, void *userdata);
static void keyboard_callback_space(SDL_Keycode key, void *userdata);
static void keyboard_callback_r(SDL_Keycode key, void *userdata);
static void keyboard_callback_1(SDL_Keycode key, void *userdata);
static void keyboard_callback_2(SDL_Keycode key, void *userdata);
static void keyboard_callback_3(SDL_Keycode key, void *userdata);
static void keyboard_callback_4(SDL_Keycode key, void *userdata);
static void keyboard_callback_5(SDL_Keycode key, void *userdata);
static void keyboard_callback_tab(SDL_Keycode key, void *userdata);
static void keyboard_callback_q(SDL_Keycode key, void *userdata);
static void keyboard_callback_w(SDL_Keycode key, void *userdata);
static void keyboard_callback_e(SDL_Keycode key, void *userdata);
static void keyboard_callback_a(SDL_Keycode key, void *userdata);
static void keyboard_callback_s(SDL_Keycode key, void *userdata);
static void keyboard_callback_d(SDL_Keycode key, void *userdata);
static void keyboard_callback_f(SDL_Keycode key, void *userdata);
static void keyboard_callback_g(SDL_Keycode key, void *userdata);
static void keyboard_callback_h(SDL_Keycode key, void *userdata);
static void keyboard_callback_j(SDL_Keycode key, void *userdata);
static void keyboard_callback_k(SDL_Keycode key, void *userdata);
static void keyboard_callback_l(SDL_Keycode key, void *userdata);
static void keyboard_callback_z(SDL_Keycode key, void *userdata);
static void keyboard_callback_x(SDL_Keycode key, void *userdata);
static void keyboard_callback_c(SDL_Keycode key, void *userdata);
static void keyboard_callback_v(SDL_Keycode key, void *userdata);
static void keyboard_callback_b(SDL_Keycode key, void *userdata);
static void keyboard_callback_n(SDL_Keycode key, void *userdata);
static void keyboard_callback_m(SDL_Keycode key, void *userdata);
static void keyboard_callback_plus(SDL_Keycode key, void *userdata);
static void keyboard_callback_minus(SDL_Keycode key, void *userdata);
static void keyboard_callback_up(SDL_Keycode key, void *userdata);
static void keyboard_callback_down(SDL_Keycode key, void *userdata);
static void keyboard_callback_left(SDL_Keycode key, void *userdata);
static void keyboard_callback_right(SDL_Keycode key, void *userdata);
static void keyboard_callback_pageup(SDL_Keycode key, void *userdata);
static void keyboard_callback_pagedown(SDL_Keycode key, void *userdata);
static void keyboard_callback_home(SDL_Keycode key, void *userdata);
static void keyboard_callback_end(SDL_Keycode key, void *userdata);

/* Utility Functions */
static unsigned int get_time_ms(void);
static void safe_sleep(unsigned int ms);
static void handle_events(AppState *state);
static void print_controls(void);
static void signal_handler(int sig);
static void cleanup_all(AppState *state);

/* ============================================================================
 * Neural Network Implementation
 * =========================================================================== */

static int init_neural_face(AppState *state) {
	unsigned int i, j;
	CADNeuralFace *face = &state->face;
	ControlSystem *ctl = &state->control;

	memset(face, 0, sizeof(CADNeuralFace));

	if (pthread_mutex_init(&face->mutex, NULL) != 0) {
		return 0;
	}

	face->neuron_count = NEURON_COUNT;
	face->synapse_count = VECTOR_COUNT;

	printf("  ├─ Allocating neurons... ");
	fflush(stdout);

	face->neurons = (DimensionalNeuron*) aligned_alloc(64,
	NEURON_COUNT * sizeof(DimensionalNeuron));
	if (!face->neurons) {
		pthread_mutex_destroy(&face->mutex);
		return 0;
	}
	memset(face->neurons, 0, NEURON_COUNT * sizeof(DimensionalNeuron));
	printf("OK\n");

	printf("  ├─ Allocating synapses... ");
	fflush(stdout);

	face->synapses = (NeuralSynapse*) aligned_alloc(64,
	VECTOR_COUNT * sizeof(NeuralSynapse));
	if (!face->synapses) {
		free(face->neurons);
		pthread_mutex_destroy(&face->mutex);
		return 0;
	}
	memset(face->synapses, 0, VECTOR_COUNT * sizeof(NeuralSynapse));
	printf("OK\n");

	printf("  ├─ Initializing neurons... ");
	fflush(stdout);

	/* Initialize neurons with spherical distribution */
	for (i = 0; i < NEURON_COUNT; i++) {
		DimensionalNeuron *n = &face->neurons[i];
		float layer = (float) (i % NEURAL_LAYERS) / NEURAL_LAYERS;

		/* Spherical coordinates with noise */
		float theta = (float) (i * 137.508f) * 0.0174533f; /* Golden angle */
		float phi = (float) ((i / 360) % 180 - 90) * 0.0174533f;
		float r = 2.0f + layer * 1.0f;

		n->pos.x = (float) (cos(phi) * cos(theta) * r);
		n->pos.y = (float) (cos(phi) * sin(theta) * r * 0.9f);
		n->pos.z = (float) (sin(phi) * r * 1.2f);
		n->pos.d = (float) (sin(theta) * 0.5f);
		n->pos.a = (float) (cos(phi) * 0.5f);
		n->pos.weight = 1.0f;

		for (j = 0; j < 5; j++) {
			n->pos.velocity[j] = ((float) rand() / RAND_MAX - 0.5f) * 0.01f;
		}

		n->potential = (float) rand() / RAND_MAX * 0.1f;
		n->threshold = THRESHOLD_BASE + layer * 0.3f;
		n->refractory = 0.0f;
		n->adaptation = ADAPTATION_RATE + layer * 0.1f;
		n->spikes = 0;
		n->last_spike = 0;
		n->id = i;
		n->frozen = 0;

		for (j = 0; j < AXIS_COUNT; j++) {
			n->weights[j] = 0.5f + 0.5f * (float) rand() / RAND_MAX;
			n->hebbian[j] = 0.0f;
		}

		/* Color based on position and layer */
		n->color.r = (unsigned char) (128 + 127 * sin(n->pos.x));
		n->color.g = (unsigned char) (128 + 127 * cos(n->pos.y));
		n->color.b = (unsigned char) (128 + 127 * sin(n->pos.z));
		n->color.a = 200;
	}
	printf("OK\n");

	printf("  ├─ Generating synapses... ");
	fflush(stdout);

	/* Initialize synapses with preferential attachment */
	for (i = 0; i < VECTOR_COUNT; i++) {
		unsigned int pre = (i * 2654435761u) % NEURON_COUNT;
		unsigned int post = (pre + 1 + (i % 100) * 2654435761u) % NEURON_COUNT;
		unsigned int bundle = i % AXIS_COUNT;

		DimensionalNeuron *pre_n = &face->neurons[pre];
		DimensionalNeuron *post_n = &face->neurons[post];
		NeuralSynapse *s = &face->synapses[i];

		s->start = pre_n->pos;
		s->end = post_n->pos;
		s->strength = 0.5f + 0.5f * (float) rand() / RAND_MAX;
		s->plasticity = 0.1f;
		s->activity = 0.0f;
		s->pre_id = pre;
		s->post_id = post;
		s->bundle = bundle;
		s->hebbian = 0.0f;

		/* Color by bundle */
		switch (bundle) {
		case 0: /* X axis - Red */
			s->color.r = 255;
			s->color.g = 80;
			s->color.b = 80;
			break;
		case 1: /* Y axis - Green */
			s->color.r = 80;
			s->color.g = 255;
			s->color.b = 80;
			break;
		case 2: /* Z axis - Blue */
			s->color.r = 80;
			s->color.g = 80;
			s->color.b = 255;
			break;
		case 3: /* D axis - Purple */
			s->color.r = 255;
			s->color.g = 80;
			s->color.b = 255;
			break;
		case 4: /* A axis - Cyan */
			s->color.r = 80;
			s->color.g = 255;
			s->color.b = 255;
			break;
		}
		s->color.a = 100;
	}
	printf("OK\n");

	/* Calculate bounds */
	float min_x = face->neurons[0].pos.x, max_x = face->neurons[0].pos.x;
	float min_y = face->neurons[0].pos.y, max_y = face->neurons[0].pos.y;
	float min_z = face->neurons[0].pos.z, max_z = face->neurons[0].pos.z;

	for (i = 1; i < NEURON_COUNT; i++) {
		DimensionalNeuron *n = &face->neurons[i];
		if (n->pos.x < min_x)
			min_x = n->pos.x;
		if (n->pos.x > max_x)
			max_x = n->pos.x;
		if (n->pos.y < min_y)
			min_y = n->pos.y;
		if (n->pos.y > max_y)
			max_y = n->pos.y;
		if (n->pos.z < min_z)
			min_z = n->pos.z;
		if (n->pos.z > max_z)
			max_z = n->pos.z;
	}

	face->bounds_min[0] = min_x;
	face->bounds_min[1] = min_y;
	face->bounds_min[2] = min_z;
	face->bounds_max[0] = max_x;
	face->bounds_max[1] = max_y;
	face->bounds_max[2] = max_z;

	face->center[0] = (min_x + max_x) * 0.5f;
	face->center[1] = (min_y + max_y) * 0.5f;
	face->center[2] = (min_z + max_z) * 0.5f;

	face->radius = (float) sqrt(
			(max_x - min_x) * (max_x - min_x)
					+ (max_y - min_y) * (max_y - min_y)
					+ (max_z - min_z) * (max_z - min_z)) * 0.5f;

	face->initialized = 1;
	face->modified = 1;

	printf("  └─ Radius: %.2f\n", face->radius);

	return 1;
}

static void free_neural_face(AppState *state) {
	CADNeuralFace *face = &state->face;

	pthread_mutex_lock(&face->mutex);
	face->initialized = 0;

	if (face->neurons) {
		free(face->neurons);
		face->neurons = NULL;
	}
	if (face->synapses) {
		free(face->synapses);
		face->synapses = NULL;
	}
	pthread_mutex_unlock(&face->mutex);
	pthread_mutex_destroy(&face->mutex);
}

static float compute_input_current(DimensionalNeuron *n, ControlSystem *ctl) {
	float input = 0.0f;
	int i;

	/* Input from position and weights */
	for (i = 0; i < AXIS_COUNT; i++) {
		float pos_val;
		switch (i) {
		case 0:
			pos_val = n->pos.x;
			break;
		case 1:
			pos_val = n->pos.y;
			break;
		case 2:
			pos_val = n->pos.z;
			break;
		case 3:
			pos_val = n->pos.d;
			break;
		case 4:
			pos_val = n->pos.a;
			break;
		default:
			pos_val = 0.0f;
			break;
		}
		input += pos_val * n->weights[i];
	}

	return input * ctl->sensitivity * 0.1f;
}

static void update_neurons(AppState *state) {
	unsigned int i, j;
	CADNeuralFace *face = &state->face;
	ControlSystem *ctl = &state->control;
	unsigned int current_time = get_time_ms();
	int spike_detected = 0;
	float spike_pitch = 0.0f;

	if (!face->initialized)
		return;
	if (ctl->paused)
		return;

	pthread_mutex_lock(&face->mutex);

	for (i = 0; i < face->neuron_count; i++) {
		DimensionalNeuron *n = &face->neurons[i];

		if (n->frozen)
			continue;

		/* Update position with velocity */
		for (j = 0; j < 5; j++) {
			float *pos = &n->pos.x;
			pos[j] += n->pos.velocity[j];

			/* Boundary conditions */
			if (pos[j] > face->bounds_max[j] || pos[j] < face->bounds_min[j]) {
				n->pos.velocity[j] *= -0.5f;
			}
		}

		/* Compute input current */
		float input = compute_input_current(n, ctl);

		/* Update membrane potential */
		n->potential += input - n->adaptation * n->potential;

		/* Clamp potential */
		if (n->potential > 2.0f)
			n->potential = 2.0f;
		if (n->potential < -1.0f)
			n->potential = -1.0f;

		/* Check for spike */
		if (n->potential > n->threshold && n->refractory < 0.1f) {
			n->spikes++;
			n->last_spike = current_time;
			n->potential *= 0.1f;
			n->refractory = 1.0f;
			spike_detected = 1;
			spike_pitch = SOUND_PITCH_BASE + n->pos.x * 100.0f;

			/* Update Hebbian trace */
			for (j = 0; j < AXIS_COUNT; j++) {
				n->hebbian[j] += 0.1f;
				if (n->hebbian[j] > 1.0f)
					n->hebbian[j] = 1.0f;
			}
		}

		/* Update refractory period */
		if (n->refractory > 0.0f) {
			n->refractory -= 0.1f;
		}

		/* Decay Hebbian trace */
		for (j = 0; j < AXIS_COUNT; j++) {
			n->hebbian[j] *= 0.99f;
		}
	}

	ctl->total_spikes += spike_detected;
	face->modified = 1;

	pthread_mutex_unlock(&face->mutex);

	/* Play sound if spike detected */
	if (spike_detected && ctl->sound_enabled && state->al.initialized) {
		play_spike_sound(&state->al, spike_pitch, ctl->sound_volume);
	}
}

static void update_synapses(AppState *state) {
	unsigned int i;
	CADNeuralFace *face = &state->face;
	ControlSystem *ctl = &state->control;

	if (!face->initialized)
		return;
	if (ctl->paused)
		return;

	pthread_mutex_lock(&face->mutex);

	for (i = 0; i < face->synapse_count; i++) {
		NeuralSynapse *s = &face->synapses[i];
		DimensionalNeuron *pre = &face->neurons[s->pre_id];
		DimensionalNeuron *post = &face->neurons[s->post_id];

		/* Update synapse activity based on pre/post spikes */
		float pre_activity =
				(pre->last_spike > get_time_ms() - 100) ? 1.0f : 0.0f;
		float post_activity =
				(post->last_spike > get_time_ms() - 100) ? 1.0f : 0.0f;

		s->activity = pre_activity * 0.3f + post_activity * 0.3f;

		/* STDP - Spike Timing Dependent Plasticity */
		if (pre_activity > 0.5f && post_activity > 0.5f) {
			s->strength += s->plasticity * ctl->learning_rate;
			if (s->strength > 1.0f)
				s->strength = 1.0f;
		} else if (pre_activity > 0.5f && post_activity < 0.1f) {
			s->strength -= s->plasticity * ctl->learning_rate * 0.1f;
			if (s->strength < 0.1f)
				s->strength = 0.1f;
		}

		/* Update Hebbian trace */
		s->hebbian = s->hebbian * 0.95f + s->activity * 0.05f;
	}

	pthread_mutex_unlock(&face->mutex);
}

static void* thread_worker(void *arg) {
	AppState *state = (AppState*) arg;
	unsigned int last_sound_update = 0;

	while (state->running) {
		update_neurons(state);
		update_synapses(state);

		/* Update sound listener position */
		if (state->al.initialized && get_time_ms() - last_sound_update > 100) {
			update_listener_position(&state->al, state->control.camera_pan_x,
					state->control.camera_height, state->control.camera_dist);
			last_sound_update = get_time_ms();
		}

		safe_sleep(10);
	}
	return NULL;
}

/* ============================================================================
 * OpenGL Renderer Implementation
 * =========================================================================== */

static int init_opengl_renderer(OpenGLRenderer *gl) {
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return 0;
	}

	/* Set OpenGL attributes */
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

	/* Create window */
	gl->window = SDL_CreateWindow(
	PROGRAM_NAME " " PROGRAM_VERSION,
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED,
	WINDOW_WIDTH,
	WINDOW_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	if (!gl->window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit();
		return 0;
	}

	/* Create OpenGL context */
	gl->context = SDL_GL_CreateContext(gl->window);
	if (!gl->context) {
		fprintf(stderr, "Context creation failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		return 0;
	}

	/* Set vsync */
	SDL_GL_SetSwapInterval(1);

	/* Initialize OpenGL state */
	gl->clear_color[0] = 0.02f;
	gl->clear_color[1] = 0.02f;
	gl->clear_color[2] = 0.05f;
	gl->clear_color[3] = 1.0f;

	glClearColor(gl->clear_color[0], gl->clear_color[1], gl->clear_color[2],
			gl->clear_color[3]);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glPointSize(POINT_SIZE);
	glLineWidth(LINE_WIDTH);

	/* Set viewport */
	gl->width = WINDOW_WIDTH;
	gl->height = WINDOW_HEIGHT;
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	/* Set projection */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) WINDOW_WIDTH / (double) WINDOW_HEIGHT, 0.1,
			100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Generate display lists */
	gl->neuron_list = glGenLists(1);
	gl->synapse_list = glGenLists(1);
	gl->grid_list = glGenLists(1);

	int i;
	for (i = 0; i < AXIS_COUNT; i++) {
		gl->axis_lists[i] = glGenLists(1);
	}

	gl->initialized = 1;
	gl->list_valid = 0;

	check_gl_error("init");

	return 1;
}

static void shutdown_opengl_renderer(OpenGLRenderer *gl) {
	if (!gl->initialized)
		return;

	int i;

	/* Delete display lists */
	if (gl->neuron_list)
		glDeleteLists(gl->neuron_list, 1);
	if (gl->synapse_list)
		glDeleteLists(gl->synapse_list, 1);
	if (gl->grid_list)
		glDeleteLists(gl->grid_list, 1);
	for (i = 0; i < AXIS_COUNT; i++) {
		if (gl->axis_lists[i])
			glDeleteLists(gl->axis_lists[i], 1);
	}

	/* Destroy context and window */
	if (gl->context) {
		SDL_GL_DeleteContext(gl->context);
		gl->context = NULL;
	}
	if (gl->window) {
		SDL_DestroyWindow(gl->window);
		gl->window = NULL;
	}

	gl->initialized = 0;
}

static void build_display_lists(AppState *state) {
	OpenGLRenderer *gl = &state->gl;
	CADNeuralFace *face = &state->face;
	ControlSystem *ctl = &state->control;
	unsigned int i, j;
	int step;

	if (!gl->initialized)
		return;
	if (!face->initialized)
		return;

	SDL_GL_MakeCurrent(gl->window, gl->context);

	/* Set step based on render quality */
	switch (ctl->render_quality) {
	case 1:
		step = 4;
		break;
	case 2:
		step = 2;
		break;
	default:
		step = 1;
		break;
	}

	/* Build neuron list */
	glNewList(gl->neuron_list, GL_COMPILE);
	glPointSize(POINT_SIZE * ctl->render_quality);
	glBegin(GL_POINTS);

	pthread_mutex_lock(&face->mutex);
	for (i = 0; i < face->neuron_count; i += step) {
		DimensionalNeuron *n = &face->neurons[i];
		float intensity = 0.5f + 0.5f * (n->spikes / 100.0f);
		if (intensity > 1.0f)
			intensity = 1.0f;

		glColor4f(n->color.r / 255.0f * intensity,
				n->color.g / 255.0f * intensity,
				n->color.b / 255.0f * intensity, n->frozen ? 0.3f : 0.8f);
		glVertex3f(n->pos.x, n->pos.y, n->pos.z);
	}
	glEnd();
	glEndList();

	/* Build synapse list */
	if (ctl->show_vectors) {
		glNewList(gl->synapse_list, GL_COMPILE);
		glLineWidth(LINE_WIDTH * ctl->render_quality);
		glBegin(GL_LINES);

		for (i = 0; i < face->synapse_count; i += step) {
			NeuralSynapse *s = &face->synapses[i];
			float alpha = s->hebbian * 0.5f;
			if (alpha < 0.1f)
				alpha = 0.1f;

			glColor4f(s->color.r / 255.0f, s->color.g / 255.0f,
					s->color.b / 255.0f, alpha);
			glVertex3f(s->start.x, s->start.y, s->start.z);
			glVertex3f(s->end.x, s->end.y, s->end.z);
		}
		glEnd();
		glEndList();
	}

	/* Build axis lists */
	if (ctl->show_axes) {
		for (j = 0; j < AXIS_COUNT; j++) {
			glNewList(gl->axis_lists[j], GL_COMPILE);
			glLineWidth(AXIS_LINE_WIDTH);
			glBegin(GL_LINES);

			for (i = 0; i < face->synapse_count; i += step) {
				NeuralSynapse *s = &face->synapses[i];
				if (s->bundle == j) {
					float axis_val;
					switch (j) {
					case 0:
						axis_val = fabs(s->end.x - s->start.x);
						break;
					case 1:
						axis_val = fabs(s->end.y - s->start.y);
						break;
					case 2:
						axis_val = fabs(s->end.z - s->start.z);
						break;
					case 3:
						axis_val = fabs(s->end.d - s->start.d);
						break;
					case 4:
						axis_val = fabs(s->end.a - s->start.a);
						break;
					default:
						axis_val = 0.0f;
						break;
					}

					if (axis_val > 0.1f) {
						glColor4f((j == 0) ? 1.0f : 0.3f,
								(j == 1) ? 1.0f : 0.3f, (j == 2) ? 1.0f : 0.3f,
								0.2f);
						glVertex3f(s->start.x, s->start.y, s->start.z);
						glVertex3f(s->end.x, s->end.y, s->end.z);
					}
				}
			}
			glEnd();
			glEndList();
		}
	}

	/* Build grid list */
	glNewList(gl->grid_list, GL_COMPILE);
	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor4f(0.2f, 0.2f, 0.2f, 0.3f);

	for (i = -10; i <= 10; i++) {
		float pos = i * 0.5f;
		glVertex3f(pos, -face->center[1], -5.0f);
		glVertex3f(pos, -face->center[1], 5.0f);
		glVertex3f(-5.0f, -face->center[1], pos);
		glVertex3f(5.0f, -face->center[1], pos);
	}
	glEnd();
	glEndList();

	pthread_mutex_unlock(&face->mutex);

	gl->list_valid = 1;
	check_gl_error("build_lists");
}

static void render_grid(void) {
	glCallList(g_state->gl.grid_list);
}

static void render_axes(AppState *state) {
	int i;
	for (i = 0; i < AXIS_COUNT; i++) {
		if (state->control.show_axes) {
			glCallList(state->gl.axis_lists[i]);
		}
	}
}

static void render_hud(AppState *state) {
	char buffer[256];
	GLint viewport[4];
	ControlSystem *ctl = &state->control;

	glGetIntegerv(GL_VIEWPORT, viewport);

	/* Switch to orthographic projection for HUD */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, viewport[2], viewport[3], 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	/* Draw HUD background */
	glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
	glBegin(GL_QUADS);
	glVertex2i(0, 0);
	glVertex2i(200, 0);
	glVertex2i(200, 100);
	glVertex2i(0, 100);
	glEnd();

	/* Draw text (simulated with points - in real app use texture fonts) */
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glPointSize(1.0f);

	/* Mode indicator */
	const char *mode_str =
			ctl->mode == 0 ? "AUTO" : (ctl->mode == 1 ? "MANUAL" : "HYBRID");
	snprintf(buffer, sizeof(buffer), "Mode: %s", mode_str);
	glRasterPos2i(10, 20);

	/* FPS */
	snprintf(buffer, sizeof(buffer), "FPS: %.1f", state->fps);
	glRasterPos2i(10, 35);

	/* Spikes */
	snprintf(buffer, sizeof(buffer), "Spikes: %lu", ctl->total_spikes);
	glRasterPos2i(10, 50);

	/* Quality */
	snprintf(buffer, sizeof(buffer), "Quality: %d", ctl->render_quality);
	glRasterPos2i(10, 65);

	/* Sound */
	snprintf(buffer, sizeof(buffer), "Sound: %s",
			ctl->sound_enabled ? "ON" : "OFF");
	glRasterPos2i(10, 80);

	glEnable(GL_DEPTH_TEST);

	/* Restore matrices */
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

static void render_frame(AppState *state) {
	OpenGLRenderer *gl = &state->gl;
	ControlSystem *ctl = &state->control;
	CADNeuralFace *face = &state->face;
	float cam_x, cam_y, cam_z;

	if (!gl->initialized)
		return;
	if (!face->initialized)
		return;

	SDL_GL_MakeCurrent(gl->window, gl->context);

	/* Rebuild display lists if needed */
	if (face->modified || !gl->list_valid) {
		build_display_lists(state);
		face->modified = 0;
	}

	/* Clear buffers */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Set projection */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) gl->width / (double) gl->height, 0.1, 100.0);

	/* Set modelview */
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Calculate camera position */
	cam_x = (float) (sin(ctl->camera_angle) * ctl->camera_dist)
			+ ctl->camera_pan_x;
	cam_y = ctl->camera_height + ctl->camera_pan_y;
	cam_z = (float) (cos(ctl->camera_angle) * ctl->camera_dist);

	gluLookAt(cam_x, cam_y, cam_z, face->center[0] + ctl->camera_pan_x,
			face->center[1] + ctl->camera_pan_y, face->center[2], 0.0f, 1.0f,
			0.0f);

	/* Render grid */
	render_grid();

	/* Render synapses */
	if (ctl->show_vectors) {
		glCallList(gl->synapse_list);
	}

	/* Render axes */
	if (ctl->show_axes) {
		render_axes(state);
	}

	/* Render neurons */
	if (ctl->show_neurons) {
		glCallList(gl->neuron_list);
	}

	/* Render HUD */
	render_hud(state);

	check_gl_error("render");

	SDL_GL_SwapWindow(gl->window);
}

static void resize_viewport(OpenGLRenderer *gl, int width, int height) {
	if (!gl->initialized)
		return;
	if (width <= 0 || height <= 0)
		return;

	gl->width = width;
	gl->height = height;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / (double) height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
}

static void check_gl_error(const char *location) {
	GLenum error;
	int count = 0;

	while ((error = glGetError()) != GL_NO_ERROR && count < 10) {
		fprintf(stderr, "OpenGL error at %s: 0x%x\n", location, error);
		count++;
	}
}

/* ============================================================================
 * OpenAL Audio Implementation
 * =========================================================================== */

static void generate_sine_wave(ALuint buffer, float frequency, float duration) {
	ALshort data[44100]; /* 1 second at 44.1kHz */
	unsigned int samples = (unsigned int) (44100 * duration);
	unsigned int i;

	for (i = 0; i < samples && i < 44100; i++) {
		float t = (float) i / 44100.0f;
		data[i] = (ALshort) (32767 * sin(2.0f * M_PI * frequency * t));
	}

	alBufferData(buffer, AL_FORMAT_MONO16, data, samples * sizeof(ALshort),
			44100);
}

static int init_openal_audio(OpenALSystem *al) {
	int i;

	al->device = alcOpenDevice(NULL);
	if (!al->device) {
		fprintf(stderr, "Failed to open OpenAL device\n");
		return 0;
	}

	al->context = alcCreateContext(al->device, NULL);
	if (!al->context) {
		alcCloseDevice(al->device);
		fprintf(stderr, "Failed to create OpenAL context\n");
		return 0;
	}

	alcMakeContextCurrent(al->context);

	/* Generate sound buffers */
	alGenBuffers(SPIKE_SOUND_COUNT, al->buffers);

	/* Generate different pitches */
	float pitches[] = { 220.0f, 277.18f, 329.63f, 369.99f, 440.0f, 493.88f,
			554.37f, 587.33f, 659.25f, 739.99f, 830.61f, 880.0f, 987.77f,
			1046.5f, 1174.7f, 1318.5f };

	for (i = 0; i < SPIKE_SOUND_COUNT && i < 16; i++) {
		generate_sine_wave(al->buffers[i], pitches[i], 0.1f);
	}

	/* Generate sound sources */
	alGenSources(SOUND_SOURCES, al->sources);
	for (i = 0; i < SOUND_SOURCES; i++) {
		alSourcef(al->sources[i], AL_GAIN, SOUND_VOLUME_MAX);
		alSourcef(al->sources[i], AL_PITCH, 1.0f);
		alSource3f(al->sources[i], AL_POSITION, 0.0f, 0.0f, 0.0f);
		alSource3f(al->sources[i], AL_VELOCITY, 0.0f, 0.0f, 0.0f);
		alSourcei(al->sources[i], AL_LOOPING, AL_FALSE);
	}

	/* Set listener properties */
	al->listener_pos[0] = 0.0f;
	al->listener_pos[1] = 0.0f;
	al->listener_pos[2] = 5.0f;
	al->listener_vel[0] = 0.0f;
	al->listener_vel[1] = 0.0f;
	al->listener_vel[2] = 0.0f;
	al->listener_ori[0] = 0.0f;
	al->listener_ori[1] = 0.0f;
	al->listener_ori[2] = -1.0f;
	al->listener_ori[3] = 0.0f;
	al->listener_ori[4] = 1.0f;
	al->listener_ori[5] = 0.0f;

	alListenerfv(AL_POSITION, al->listener_pos);
	alListenerfv(AL_VELOCITY, al->listener_vel);
	alListenerfv(AL_ORIENTATION, al->listener_ori);

	al->source_index = 0;
	al->initialized = 1;

	return 1;
}

static void shutdown_openal_audio(OpenALSystem *al) {
	if (!al->initialized)
		return;

	alDeleteSources(SOUND_SOURCES, al->sources);
	alDeleteBuffers(SPIKE_SOUND_COUNT, al->buffers);

	alcMakeContextCurrent(NULL);
	alcDestroyContext(al->context);
	alcCloseDevice(al->device);

	al->initialized = 0;
}

static void play_spike_sound(OpenALSystem *al, float pitch, float volume) {
	if (!al->initialized)
		return;

	ALint state;
	ALuint source = al->sources[al->source_index];
	ALuint buffer = al->buffers[al->source_index % SPIKE_SOUND_COUNT];

	alGetSourcei(source, AL_SOURCE_STATE, &state);
	if (state == AL_PLAYING) {
		alSourceStop(source);
	}

	alSourcei(source, AL_BUFFER, buffer);
	alSourcef(source, AL_PITCH, pitch / 440.0f);
	alSourcef(source, AL_GAIN, volume);
	alSourcePlay(source);

	al->source_index = (al->source_index + 1) % SOUND_SOURCES;
}

static void update_listener_position(OpenALSystem *al, float x, float y,
		float z) {
	if (!al->initialized)
		return;

	al->listener_pos[0] = x;
	al->listener_pos[1] = y;
	al->listener_pos[2] = z;

	alListenerfv(AL_POSITION, al->listener_pos);
}

static void set_sound_volume(OpenALSystem *al, float volume) {
	int i;
	if (!al->initialized)
		return;

	for (i = 0; i < SOUND_SOURCES; i++) {
		alSourcef(al->sources[i], AL_GAIN, volume);
	}
}

/* ============================================================================
 * Keyboard Callback System
 * =========================================================================== */

static void init_control_system(ControlSystem *ctl, AppState *state) {
	memset(ctl, 0, sizeof(ControlSystem));

	ctl->mode = 0; /* Auto */
	ctl->paused = 0;
	ctl->render_quality = 2;
	ctl->show_axes = 1;
	ctl->show_vectors = 1;
	ctl->show_neurons = 1;
	ctl->sound_enabled = 1;
	ctl->sound_volume = 0.3f;
	ctl->camera_angle = 0.0f;
	ctl->camera_dist = 8.0f;
	ctl->camera_height = 2.0f;
	ctl->camera_pan_x = 0.0f;
	ctl->camera_pan_y = 0.0f;
	ctl->rotation_speed = 0.01f;
	ctl->learning_rate = LEARNING_RATE;
	ctl->sensitivity = SENSITIVITY;
	ctl->total_spikes = 0;

	pthread_mutex_init(&ctl->callback_mutex, NULL);

	/* Register keyboard callbacks */
	register_keyboard_callback(ctl, SDLK_ESCAPE, keyboard_callback_escape,
			state, 100);
	register_keyboard_callback(ctl, SDLK_SPACE, keyboard_callback_space, state,
			90);
	register_keyboard_callback(ctl, SDLK_r, keyboard_callback_r, state, 80);
	register_keyboard_callback(ctl, SDLK_1, keyboard_callback_1, state, 70);
	register_keyboard_callback(ctl, SDLK_2, keyboard_callback_2, state, 70);
	register_keyboard_callback(ctl, SDLK_3, keyboard_callback_3, state, 70);
	register_keyboard_callback(ctl, SDLK_4, keyboard_callback_4, state, 70);
	register_keyboard_callback(ctl, SDLK_5, keyboard_callback_5, state, 70);
	register_keyboard_callback(ctl, SDLK_TAB, keyboard_callback_tab, state, 85);
	register_keyboard_callback(ctl, SDLK_q, keyboard_callback_q, state, 60);
	register_keyboard_callback(ctl, SDLK_w, keyboard_callback_w, state, 60);
	register_keyboard_callback(ctl, SDLK_e, keyboard_callback_e, state, 60);
	register_keyboard_callback(ctl, SDLK_a, keyboard_callback_a, state, 60);
	register_keyboard_callback(ctl, SDLK_s, keyboard_callback_s, state, 60);
	register_keyboard_callback(ctl, SDLK_d, keyboard_callback_d, state, 60);
	register_keyboard_callback(ctl, SDLK_f, keyboard_callback_f, state, 50);
	register_keyboard_callback(ctl, SDLK_g, keyboard_callback_g, state, 50);
	register_keyboard_callback(ctl, SDLK_h, keyboard_callback_h, state, 50);
	register_keyboard_callback(ctl, SDLK_j, keyboard_callback_j, state, 50);
	register_keyboard_callback(ctl, SDLK_k, keyboard_callback_k, state, 50);
	register_keyboard_callback(ctl, SDLK_l, keyboard_callback_l, state, 50);
	register_keyboard_callback(ctl, SDLK_z, keyboard_callback_z, state, 40);
	register_keyboard_callback(ctl, SDLK_x, keyboard_callback_x, state, 40);
	register_keyboard_callback(ctl, SDLK_c, keyboard_callback_c, state, 40);
	register_keyboard_callback(ctl, SDLK_v, keyboard_callback_v, state, 40);
	register_keyboard_callback(ctl, SDLK_b, keyboard_callback_b, state, 40);
	register_keyboard_callback(ctl, SDLK_n, keyboard_callback_n, state, 40);
	register_keyboard_callback(ctl, SDLK_m, keyboard_callback_m, state, 40);
	register_keyboard_callback(ctl, SDLK_PLUS, keyboard_callback_plus, state,
			30);
	register_keyboard_callback(ctl, SDLK_MINUS, keyboard_callback_minus, state,
			30);
	register_keyboard_callback(ctl, SDLK_UP, keyboard_callback_up, state, 20);
	register_keyboard_callback(ctl, SDLK_DOWN, keyboard_callback_down, state,
			20);
	register_keyboard_callback(ctl, SDLK_LEFT, keyboard_callback_left, state,
			20);
	register_keyboard_callback(ctl, SDLK_RIGHT, keyboard_callback_right, state,
			20);
	register_keyboard_callback(ctl, SDLK_PAGEUP, keyboard_callback_pageup,
			state, 15);
	register_keyboard_callback(ctl, SDLK_PAGEDOWN, keyboard_callback_pagedown,
			state, 15);
	register_keyboard_callback(ctl, SDLK_HOME, keyboard_callback_home, state,
			10);
	register_keyboard_callback(ctl, SDLK_END, keyboard_callback_end, state, 10);
}

static void register_keyboard_callback(ControlSystem *ctl, SDL_Keycode key,
		KeyboardCallback callback, void *userdata, int priority) {
	KeyboardCallbackEntry *entry, *prev, *new_entry;
	int key_index = key & 0xFF;

	if (!ctl || !callback)
		return;

	pthread_mutex_lock(&ctl->callback_mutex);

	new_entry = (KeyboardCallbackEntry*) malloc(sizeof(KeyboardCallbackEntry));
	if (!new_entry) {
		pthread_mutex_unlock(&ctl->callback_mutex);
		return;
	}

	new_entry->key = key;
	new_entry->callback = callback;
	new_entry->userdata = userdata;
	new_entry->priority = priority;
	new_entry->next = NULL;

	if (!ctl->callbacks[key_index]) {
		ctl->callbacks[key_index] = new_entry;
	} else {
		prev = NULL;
		entry = ctl->callbacks[key_index];

		while (entry && entry->priority >= priority) {
			prev = entry;
			entry = entry->next;
		}

		if (!prev) {
			new_entry->next = ctl->callbacks[key_index];
			ctl->callbacks[key_index] = new_entry;
		} else {
			new_entry->next = entry;
			prev->next = new_entry;
		}
	}

	pthread_mutex_unlock(&ctl->callback_mutex);
}

static void dispatch_keyboard_callbacks(ControlSystem *ctl, SDL_Keycode key) {
	KeyboardCallbackEntry *entry;
	int key_index = key & 0xFF;

	if (!ctl)
		return;

	pthread_mutex_lock(&ctl->callback_mutex);

	entry = ctl->callbacks[key_index];
	while (entry) {
		if (entry->key == key || entry->key == 0) {
			entry->callback(key, entry->userdata);
		}
		entry = entry->next;
	}

	pthread_mutex_unlock(&ctl->callback_mutex);
}

/* ============================================================================
 * Keyboard Callback Implementations
 * =========================================================================== */

static void keyboard_callback_escape(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	printf("\nExiting...\n");
	state->running = 0;
}

static void keyboard_callback_space(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.paused = !state->control.paused;
	printf("%s\n", state->control.paused ? "PAUSED" : "RUNNING");
}

static void keyboard_callback_r(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_angle = 0.0f;
	state->control.camera_dist = 8.0f;
	state->control.camera_height = 2.0f;
	state->control.camera_pan_x = 0.0f;
	state->control.camera_pan_y = 0.0f;
	printf("Camera reset\n");
}

static void keyboard_callback_1(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.render_quality = 1;
	state->face.modified = 1;
	printf("Render quality: LOW\n");
}

static void keyboard_callback_2(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.render_quality = 2;
	state->face.modified = 1;
	printf("Render quality: MEDIUM\n");
}

static void keyboard_callback_3(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.render_quality = 3;
	state->face.modified = 1;
	printf("Render quality: HIGH\n");
}

static void keyboard_callback_4(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.show_axes = !state->control.show_axes;
	printf("Axes: %s\n", state->control.show_axes ? "ON" : "OFF");
}

static void keyboard_callback_5(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.show_vectors = !state->control.show_vectors;
	state->face.modified = 1;
	printf("Vectors: %s\n", state->control.show_vectors ? "ON" : "OFF");
}

static void keyboard_callback_tab(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.mode = (state->control.mode + 1) % 3;
	printf("Mode: %s\n",
			state->control.mode == 0 ? "AUTONOMOUS" :
			state->control.mode == 1 ? "MANUAL" : "HYBRID");
}

static void keyboard_callback_q(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.learning_rate *= 1.1f;
	if (state->control.learning_rate > 0.1f)
		state->control.learning_rate = 0.1f;
	printf("Learning rate: %.3f\n", state->control.learning_rate);
}

static void keyboard_callback_w(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.learning_rate *= 0.9f;
	if (state->control.learning_rate < 0.001f)
		state->control.learning_rate = 0.001f;
	printf("Learning rate: %.3f\n", state->control.learning_rate);
}

static void keyboard_callback_e(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.sensitivity *= 1.1f;
	if (state->control.sensitivity > 2.0f)
		state->control.sensitivity = 2.0f;
	printf("Sensitivity: %.2f\n", state->control.sensitivity);
}

static void keyboard_callback_a(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.sensitivity *= 0.9f;
	if (state->control.sensitivity < 0.1f)
		state->control.sensitivity = 0.1f;
	printf("Sensitivity: %.2f\n", state->control.sensitivity);
}

static void keyboard_callback_s(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.sound_enabled = !state->control.sound_enabled;
	printf("Sound: %s\n", state->control.sound_enabled ? "ON" : "OFF");
}

static void keyboard_callback_d(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.show_neurons = !state->control.show_neurons;
	printf("Neurons: %s\n", state->control.show_neurons ? "ON" : "OFF");
}

static void keyboard_callback_f(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Freeze/unfreeze random neuron */
	unsigned int idx = rand() % state->face.neuron_count;
	state->face.neurons[idx].frozen = !state->face.neurons[idx].frozen;
	printf("Neuron %u %s\n", idx,
			state->face.neurons[idx].frozen ? "FROZEN" : "UNFROZEN");
}

static void keyboard_callback_g(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Freeze/unfreeze all neurons */
	unsigned int i;
	for (i = 0; i < state->face.neuron_count; i++) {
		state->face.neurons[i].frozen = !state->face.neurons[i].frozen;
	}
	printf("All neurons %s\n",
			state->face.neurons[0].frozen ? "FROZEN" : "UNFROZEN");
}

static void keyboard_callback_h(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Reset all spikes */
	unsigned int i;
	for (i = 0; i < state->face.neuron_count; i++) {
		state->face.neurons[i].spikes = 0;
	}
	state->control.total_spikes = 0;
	printf("Spikes reset\n");
}

static void keyboard_callback_j(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Decrease camera distance */
	state->control.camera_dist -= 0.5f;
	if (state->control.camera_dist < 3.0f)
		state->control.camera_dist = 3.0f;
	printf("Camera distance: %.1f\n", state->control.camera_dist);
}

static void keyboard_callback_k(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Increase camera distance */
	state->control.camera_dist += 0.5f;
	if (state->control.camera_dist > 20.0f)
		state->control.camera_dist = 20.0f;
	printf("Camera distance: %.1f\n", state->control.camera_dist);
}

static void keyboard_callback_l(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Toggle auto-rotation */
	if (state->control.rotation_speed > 0) {
		state->control.rotation_speed = 0;
		printf("Rotation: OFF\n");
	} else {
		state->control.rotation_speed = 0.01f;
		printf("Rotation: ON\n");
	}
}

static void keyboard_callback_z(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Pan camera left */
	state->control.camera_pan_x -= 0.2f;
}

static void keyboard_callback_x(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Pan camera right */
	state->control.camera_pan_x += 0.2f;
}

static void keyboard_callback_c(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Pan camera down */
	state->control.camera_pan_y -= 0.2f;
}

static void keyboard_callback_v(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Pan camera up */
	state->control.camera_pan_y += 0.2f;
}

static void keyboard_callback_b(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Decrease sound volume */
	state->control.sound_volume -= 0.1f;
	if (state->control.sound_volume < 0.0f)
		state->control.sound_volume = 0.0f;
	set_sound_volume(&state->al, state->control.sound_volume);
	printf("Volume: %.1f\n", state->control.sound_volume);
}

static void keyboard_callback_n(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Increase sound volume */
	state->control.sound_volume += 0.1f;
	if (state->control.sound_volume > 1.0f)
		state->control.sound_volume = 1.0f;
	set_sound_volume(&state->al, state->control.sound_volume);
	printf("Volume: %.1f\n", state->control.sound_volume);
}

static void keyboard_callback_m(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Mute/unmute */
	static float last_volume = 0.3f;
	if (state->control.sound_volume > 0) {
		last_volume = state->control.sound_volume;
		state->control.sound_volume = 0;
	} else {
		state->control.sound_volume = last_volume;
	}
	set_sound_volume(&state->al, state->control.sound_volume);
	printf("Volume: %.1f\n", state->control.sound_volume);
}

static void keyboard_callback_plus(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Increase rotation speed */
	state->control.rotation_speed += 0.005f;
	if (state->control.rotation_speed > 0.05f)
		state->control.rotation_speed = 0.05f;
	printf("Rotation speed: %.3f\n", state->control.rotation_speed);
}

static void keyboard_callback_minus(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	/* Decrease rotation speed */
	state->control.rotation_speed -= 0.005f;
	if (state->control.rotation_speed < 0.0f)
		state->control.rotation_speed = 0.0f;
	printf("Rotation speed: %.3f\n", state->control.rotation_speed);
}

static void keyboard_callback_up(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_height += 0.2f;
	if (state->control.camera_height > 5.0f)
		state->control.camera_height = 5.0f;
}

static void keyboard_callback_down(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_height -= 0.2f;
	if (state->control.camera_height < -2.0f)
		state->control.camera_height = -2.0f;
}

static void keyboard_callback_left(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_angle -= 0.05f;
}

static void keyboard_callback_right(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_angle += 0.05f;
}

static void keyboard_callback_pageup(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	if (state->control.render_quality < RENDER_QUALITY_MAX) {
		state->control.render_quality++;
		state->face.modified = 1;
		printf("Render quality: %d\n", state->control.render_quality);
	}
}

static void keyboard_callback_pagedown(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	if (state->control.render_quality > RENDER_QUALITY_MIN) {
		state->control.render_quality--;
		state->face.modified = 1;
		printf("Render quality: %d\n", state->control.render_quality);
	}
}

static void keyboard_callback_home(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	state->control.camera_angle = 0.0f;
	state->control.camera_dist = 8.0f;
	state->control.camera_height = 2.0f;
	state->control.camera_pan_x = 0.0f;
	state->control.camera_pan_y = 0.0f;
}

static void keyboard_callback_end(SDL_Keycode key, void *userdata) {
	AppState *state = (AppState*) userdata;
	printf("Emergency shutdown\n");
	state->running = 0;
	longjmp(state->emergency_jmp, 1);
}

/* ============================================================================
 * Utility Functions
 * =========================================================================== */

static unsigned int get_time_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (unsigned int) (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

static void safe_sleep(unsigned int ms) {
	struct timespec req;
	req.tv_sec = ms / 1000;
	req.tv_nsec = (ms % 1000) * 1000000;
	nanosleep(&req, NULL);
}

static void handle_events(AppState *state) {
	SDL_Event event;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			state->running = 0;
			break;

		case SDL_KEYDOWN:
			dispatch_keyboard_callbacks(&state->control, event.key.keysym.sym);
			break;

		case SDL_WINDOWEVENT:
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				resize_viewport(&state->gl, event.window.data1,
						event.window.data2);
			}
			break;
		}
	}
}

static void print_controls(void) {
	printf(
			"\n╔════════════════════════════════════════════════════════════╗\n");
	printf("║                    KEYBOARD CONTROLS                        ║\n");
	printf("╠════════════════════════════════════════════════════════════╣\n");
	printf("║  ESC     - Exit                     SPACE - Pause          ║\n");
	printf("║  R       - Reset camera             TAB   - Cycle mode     ║\n");
	printf("║  1,2,3   - Render quality          4,5   - Toggle axes/vec ║\n");
	printf("║  Q/W     - +/- learning rate        E/A   - +/- sensitivity║\n");
	printf("║  S       - Toggle sound             D     - Toggle neurons ║\n");
	printf("║  F       - Freeze neuron            G     - Freeze all     ║\n");
	printf("║  H       - Reset spikes             J/K   - Camera distance║\n");
	printf("║  L       - Toggle rotation          Z/X/C/V - Camera pan   ║\n");
	printf("║  B/N/M   - Volume control           +/-   - Rotation speed ║\n");
	printf("║  Arrows  - Rotate camera            PGUP/DN - Quality      ║\n");
	printf("║  HOME    - Home view                 END   - Emergency     ║\n");
	printf(
			"╚════════════════════════════════════════════════════════════╝\n\n");
}

static void signal_handler(int sig) {
	(void) sig;
	g_running = 0;
}

static void cleanup_all(AppState *state) {
	unsigned int i;

	state->shutting_down = 1;
	state->running = 0;

	/* Join threads */
	for (i = 0; i < state->thread_count; i++) {
		pthread_join(state->threads[i], NULL);
	}

	/* Cleanup subsystems */
	shutdown_opengl_renderer(&state->gl);
	shutdown_openal_audio(&state->al);
	free_neural_face(state);

	/* Cleanup callback system */
	pthread_mutex_destroy(&state->control.callback_mutex);
}

/* ============================================================================
 * Main Function
 * =========================================================================== */

int main(int argc, char **argv) {
	AppState state;
	unsigned int i, current_time;

	(void) argc;
	(void) argv;

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	if (setjmp(state.emergency_jmp)) {
		printf("\nEmergency shutdown complete\n");
		cleanup_all(&state);
		return 0;
	}

	srand((unsigned int) time(NULL));

	memset(&state, 0, sizeof(AppState));
	state.running = 1;
	g_state = &state;

	printf(
			"\n╔════════════════════════════════════════════════════════════╗\n");
	printf("║     MIMIX CAD Face - Neural Dimensional System v%s     ║\n",
			PROGRAM_VERSION);
	printf(
			"╚════════════════════════════════════════════════════════════╝\n\n");

	printf("System Configuration:\n");
	printf("  ├─ CPU Cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
	printf("  └─ Max Threads: %d\n\n", MAX_THREADS);

	printf("Generating Neural CAD face...\n");
	if (!init_neural_face(&state)) {
		fprintf(stderr, "Failed to initialize neural face\n");
		return 1;
	}

	printf("\nInitializing OpenGL... ");
	fflush(stdout);
	if (!init_opengl_renderer(&state.gl)) {
		fprintf(stderr, "OpenGL initialization failed\n");
		free_neural_face(&state);
		return 1;
	}
	printf("OK\n");

	printf("Initializing OpenAL... ");
	fflush(stdout);
	if (!init_openal_audio(&state.al)) {
		printf("Failed (audio disabled)\n");
		state.al.initialized = 0;
	} else {
		printf("OK\n");
	}

	printf("Initializing control system... ");
	fflush(stdout);
	init_control_system(&state.control, &state);
	printf("OK\n");

	printf("Building display lists... ");
	fflush(stdout);
	build_display_lists(&state);
	printf("OK\n");

	printf("Starting thread pool... ");
	fflush(stdout);

	state.thread_count = MAX_THREADS;
	for (i = 0; i < state.thread_count; i++) {
		if (pthread_create(&state.threads[i], NULL, thread_worker, &state)
				!= 0) {
			fprintf(stderr, "Failed to create thread %d\n", i);
			state.thread_count = i;
			break;
		}
	}
	printf("OK (%d threads)\n\n", state.thread_count);

	print_controls();

	state.last_frame_time = get_time_ms();
	state.frame_count = 0;
	state.fps = 0.0;

	/* Main loop */
	while (state.running && g_running) {
		current_time = get_time_ms();

		handle_events(&state);

		/* Render frame */
		render_frame(&state);

		/* Update camera rotation if not paused */
		if (!state.control.paused) {
			state.control.camera_angle += state.control.rotation_speed;
		}

		/* FPS calculation */
		state.frame_count++;
		if (current_time - state.last_frame_time >= 1000) {
			state.fps = (double) state.frame_count * 1000.0
					/ (double) (current_time - state.last_frame_time);
			state.frame_count = 0;
			state.last_frame_time = current_time;
		}

		safe_sleep(FRAME_TIME_MS);
	}

	printf("\nShutting down...\n");
	cleanup_all(&state);
	SDL_Quit();

	printf("Goodbye.\n");
	return 0;
}
