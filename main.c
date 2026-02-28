/**
 * @file mimix/src/main.c
 * @version 4.0.7
 * @license GNU 3
 *
 * @title MIMIX CAD Face - Neural Dimensional System
 * @description Final fix for memory management - no more crashes
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

/* NUMA headers */
#define _GNU_SOURCE
#include <numa.h>
#include <numaif.h>

/* OpenCL */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

/* OpenGL */
#include <GL/gl.h>
#include <GL/glu.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* SIMD Headers */
#include <immintrin.h>
#include <x86intrin.h>

/* ============================================================================
 * Compile-time Configuration
 * =========================================================================== */

#define PROGRAM_NAME         "MIMIX CAD Face"
#define PROGRAM_VERSION      "4.0.7"
#define WINDOW_WIDTH         1280
#define WINDOW_HEIGHT        720
#define MAX_THREADS          8
#define CACHE_LINE_SIZE      64
#define SIMD_ALIGNMENT       32
#define AXIS_COUNT           5
#define VECTOR_COUNT         65536
#define NEURON_COUNT         262144
#define NEURAL_LAYERS        12
#define SYNAPSE_DENSITY      0.15f
#define MAX_SPIKE_HISTORY    64
#define POINT_SIZE           2.0f
#define LINE_WIDTH           1.0f
#define FRAME_TIME_MS        33
#define SHUTDOWN_TIMEOUT_MS  1000

/* Memory alignment macros */
#define ALIGNED(align) __attribute__((aligned(align)))
#define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#define SIMD_ALIGNED __attribute__((aligned(SIMD_ALIGNMENT)))
#define PACKED __attribute__((packed))

/* ============================================================================
 * Core Data Structures
 * =========================================================================== */

typedef struct SIMD_ALIGNED NeuralVector5D {
	float x;
	float y;
	float z;
	float d;
	float a;
	float weight;
	float learning_rate;
	float spike_rate;
} NeuralVector5D;

typedef struct PACKED ColorBGRA {
	unsigned char b;
	unsigned char g;
	unsigned char r;
	unsigned char a;
} ColorBGRA;

typedef struct CACHE_ALIGNED DimensionalNeuron {
	NeuralVector5D position;
	float membrane_potential;
	float firing_threshold;
	float refractory_period;
	float adaptation_rate;
	float synaptic_weight[AXIS_COUNT];
	unsigned int connection_count;
	unsigned int layer_id;
	ColorBGRA color;
	unsigned int id;
	unsigned int spike_count;
	float learning_rate;
	unsigned int firing_history[MAX_SPIKE_HISTORY];
	unsigned int history_index;
} DimensionalNeuron;

typedef struct CACHE_ALIGNED NeuralVectorLine {
	NeuralVector5D start;
	NeuralVector5D end;
	NeuralVector5D direction;
	float magnitude;
	float synaptic_strength;
	float plasticity;
	float firing_rate;
	unsigned int presynaptic_id;
	unsigned int postsynaptic_id;
	unsigned int axon_bundle;
	ColorBGRA color;
} NeuralVectorLine;

typedef struct CACHE_ALIGNED CADNeuralFace {
	DimensionalNeuron *neurons;
	NeuralVectorLine *vectors;
	float neural_activity[AXIS_COUNT];
	float learning_rate;
	unsigned int neuron_count;
	unsigned int vector_count;
	unsigned int active_connections;
	float bounding_box_min[AXIS_COUNT];
	float bounding_box_max[AXIS_COUNT];
	float scale;
	double timestamp;
	float center[3];
	float radius;
	pthread_mutex_t data_mutex;
	volatile int initialized;
} CADNeuralFace;

typedef struct CACHE_ALIGNED ThreadWork {
	unsigned int thread_id;
	int node_id;
	int cpu_core;
	pthread_t thread;
	cpu_set_t cpu_affinity;
	CADNeuralFace *face;
	unsigned int neuron_start;
	unsigned int neuron_end;
	unsigned int vector_start;
	unsigned int vector_end;
	float axis_weights[AXIS_COUNT];
	volatile int completed;
	volatile unsigned long spike_count;
	double processing_time;
	unsigned long long simd_ops;
	volatile int running;
	volatile int should_stop;
	pthread_mutex_t work_mutex;
} ThreadWork;

typedef struct CLContext {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel_neurons;
	cl_kernel kernel_vectors;
	cl_mem neurons_buffer;
	cl_mem vectors_buffer;
	size_t global_work_size;
	size_t local_work_size;
	volatile int initialized;
	char device_name[256];
} CLContext;

typedef struct GLContext {
	SDL_Window *window;
	SDL_GLContext gl_context;
	GLuint neuron_list;
	GLuint vector_list;
	GLuint axis_display_list[AXIS_COUNT];
	volatile int initialized;
	int window_width;
	int window_height;
	float clear_color[4];
	volatile int render_ready;
} GLContext;

typedef struct CACHE_ALIGNED AppState {
	CADNeuralFace face; /* Embedded, not a pointer */
	ThreadWork threads[MAX_THREADS];
	unsigned int thread_count;
	CLContext cl;
	GLContext gl;
	ALCdevice *al_device;
	ALCcontext *al_context;
	float axis_activity[AXIS_COUNT];
	unsigned long total_spikes;
	unsigned long long total_simd_ops;
	volatile int running;
	volatile int paused;
	volatile int shutting_down;
	double fps;
	struct timespec last_frame;
	struct timespec start_time;
	float camera_angle;
	float camera_distance;
	float camera_height;
	float rotation_speed;
	pthread_mutex_t render_mutex;
	pthread_mutex_t state_mutex;
	pthread_mutex_t cleanup_mutex;
	unsigned int last_frame_time;
	int frame_count;
	int numa_nodes;
	int cpu_cores;
	int active_connections;
} AppState;

/* ============================================================================
 * Function Prototypes
 * =========================================================================== */

static void* thread_process_neurons(void *arg);
static int init_neural_thread_pool(AppState *state);
static void wait_for_threads(AppState *state);
static void join_threads(AppState *state);
static void stop_threads(AppState *state);

static void neural_update_spike(DimensionalNeuron *neuron, float input_current);
static void process_5axis_neural_field(CADNeuralFace *face,
		unsigned int axis_id);

static int init_neural_opengl(GLContext *gl);
static void build_neural_display_lists(AppState *state);
static void render_neural_cad_face(AppState *state);
static void check_gl_error(const char *operation);
static void resize_window(GLContext *gl, int width, int height);
static void cleanup_opengl(GLContext *gl);
static int validate_gl_state(GLContext *gl);

static int init_neural_opencl(CLContext *cl);
static void cleanup_opencl(CLContext *cl);

static int init_openal(AppState *state);
static void cleanup_openal(AppState *state);

static NeuralVector5D neural_vector_normalize(const NeuralVector5D v);
static float neural_vector_length(const NeuralVector5D v);
static NeuralVector5D neural_vector_sub(const NeuralVector5D a,
		const NeuralVector5D b);
static NeuralVector5D neural_vector_add(const NeuralVector5D a,
		const NeuralVector5D b);
static float neural_vector_dot(const NeuralVector5D a, const NeuralVector5D b);

static int init_neural_face(AppState *state);
static void generate_neural_vectors(AppState *state);
static void calculate_face_bounds(AppState *state);
static void free_neural_face(AppState *state);
static NeuralVectorLine generate_neural_vector(unsigned int id,
		const DimensionalNeuron *pre, const DimensionalNeuron *post,
		unsigned int bundle_id);
static int validate_face(CADNeuralFace *face);

static unsigned int get_time_ms(void);
static void handle_events(AppState *state);
static void print_system_info(AppState *state);
static void signal_handler(int sig);
static void safe_sleep(unsigned int ms);
static void cleanup_all(AppState *state);

/* ============================================================================
 * Vector Operations Implementation
 * =========================================================================== */

static NeuralVector5D neural_vector_normalize(const NeuralVector5D v) {
	float len = (float) sqrt(
			(double) (v.x * v.x + v.y * v.y + v.z * v.z + v.d * v.d + v.a * v.a
					+ v.weight * v.weight));
	NeuralVector5D result;

	if (len > 1.0e-6f) {
		result.x = v.x / len;
		result.y = v.y / len;
		result.z = v.z / len;
		result.d = v.d / len;
		result.a = v.a / len;
		result.weight = v.weight / len;
	} else {
		result.x = result.y = result.z = result.d = result.a = result.weight =
				0.0f;
	}
	result.learning_rate = v.learning_rate;
	result.spike_rate = v.spike_rate;

	return result;
}

static float neural_vector_length(const NeuralVector5D v) {
	return (float) sqrt(
			(double) (v.x * v.x + v.y * v.y + v.z * v.z + v.d * v.d + v.a * v.a
					+ v.weight * v.weight));
}

static NeuralVector5D neural_vector_sub(const NeuralVector5D a,
		const NeuralVector5D b) {
	NeuralVector5D result;
	result.x = a.x - b.x;
	result.y = a.y - b.y;
	result.z = a.z - b.z;
	result.d = a.d - b.d;
	result.a = a.a - b.a;
	result.weight = a.weight - b.weight;
	result.learning_rate = (a.learning_rate + b.learning_rate) * 0.5f;
	result.spike_rate = (a.spike_rate + b.spike_rate) * 0.5f;
	return result;
}

static NeuralVector5D neural_vector_add(const NeuralVector5D a,
		const NeuralVector5D b) {
	NeuralVector5D result;
	result.x = a.x + b.x;
	result.y = a.y + b.y;
	result.z = a.z + b.z;
	result.d = a.d + b.d;
	result.a = a.a + b.a;
	result.weight = a.weight + b.weight;
	result.learning_rate = (a.learning_rate + b.learning_rate) * 0.5f;
	result.spike_rate = (a.spike_rate + b.spike_rate) * 0.5f;
	return result;
}

static float neural_vector_dot(const NeuralVector5D a, const NeuralVector5D b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.d * b.d + a.a * b.a
			+ a.weight * b.weight;
}

/* ============================================================================
 * Validate Face
 * =========================================================================== */

static int validate_face(CADNeuralFace *face) {
	if (!face)
		return 0;
	if (!face->initialized)
		return 0;
	if (!face->neurons)
		return 0;
	if (!face->vectors)
		return 0;
	if (face->neuron_count == 0 || face->neuron_count > NEURON_COUNT)
		return 0;
	if (face->vector_count == 0 || face->vector_count > VECTOR_COUNT)
		return 0;
	return 1;
}

/* ============================================================================
 * Neural Update Spike
 * =========================================================================== */

static void neural_update_spike(DimensionalNeuron *neuron, float input_current) {
	float activation;
	unsigned int i;

	if (!neuron)
		return;

	neuron->membrane_potential += input_current
			- (neuron->adaptation_rate * neuron->membrane_potential);

	if (neuron->membrane_potential > 2.0f)
		neuron->membrane_potential = 2.0f;
	if (neuron->membrane_potential < -1.0f)
		neuron->membrane_potential = -1.0f;

	activation = 1.0f
			/ (1.0f
					+ expf(
							-(neuron->membrane_potential
									- neuron->firing_threshold) * 5.0f));

	if (activation > 0.5f && neuron->refractory_period < 0.1f) {
		if (neuron->history_index < MAX_SPIKE_HISTORY) {
			neuron->firing_history[neuron->history_index] = 1;
		}
		neuron->spike_count++;
		if (neuron->spike_count > 1000000)
			neuron->spike_count = 1000000;
		neuron->membrane_potential *= 0.1f;
		neuron->refractory_period = 1.0f;

		for (i = 0; i < AXIS_COUNT; i++) {
			neuron->synaptic_weight[i] += neuron->learning_rate * activation;
			if (neuron->synaptic_weight[i] > 1.0f)
				neuron->synaptic_weight[i] = 1.0f;
			if (neuron->synaptic_weight[i] < 0.0f)
				neuron->synaptic_weight[i] = 0.0f;
		}
	} else {
		if (neuron->history_index < MAX_SPIKE_HISTORY) {
			neuron->firing_history[neuron->history_index] = 0;
		}
	}

	if (neuron->refractory_period > 0.0f) {
		neuron->refractory_period -= 0.1f;
		if (neuron->refractory_period < 0.0f)
			neuron->refractory_period = 0.0f;
	}

	neuron->history_index++;
	if (neuron->history_index >= MAX_SPIKE_HISTORY) {
		neuron->history_index = 0;
	}
}

/* ============================================================================
 * AVX2 Optimized Functions (if available)
 * =========================================================================== */

#ifdef __AVX2__
static void avx2_process_neurons_batch(DimensionalNeuron *neurons,
                                        const float *input_currents,
                                        unsigned int count)
{
    unsigned int i;
    for (i = 0; i + 7 < count && i < count; i += 8) {
        __m256 mem_pot = _mm256_load_ps(&neurons[i].membrane_potential);
        __m256 currents = _mm256_load_ps(&input_currents[i]);
        __m256 adaptation = _mm256_load_ps(&neurons[i].adaptation_rate);

        __m256 decay = _mm256_mul_ps(adaptation, mem_pot);
        __m256 updated = _mm256_add_ps(_mm256_sub_ps(mem_pot, decay), currents);

        _mm256_store_ps(&neurons[i].membrane_potential, updated);
    }
}
#endif

/* ============================================================================
 * 5-Axis Neural Field Processing
 * =========================================================================== */

static void process_5axis_neural_field(CADNeuralFace *face,
		unsigned int axis_id) {
	unsigned int i;
	float field_strength = 0.0f;
	float *axis_ptr = NULL;

	if (!face || !face->initialized)
		return;
	if (!face->neurons || face->neuron_count == 0)
		return;

	if (pthread_mutex_trylock(&face->data_mutex) != 0) {
		return;
	}

	for (i = 0; i < face->neuron_count; i++) {
		DimensionalNeuron *neuron = &face->neurons[i];

		switch (axis_id) {
		case 0:
			axis_ptr = &neuron->position.x;
			break;
		case 1:
			axis_ptr = &neuron->position.y;
			break;
		case 2:
			axis_ptr = &neuron->position.z;
			break;
		case 3:
			axis_ptr = &neuron->position.d;
			break;
		case 4:
			axis_ptr = &neuron->position.a;
			break;
		default:
			axis_ptr = &neuron->position.x;
			break;
		}

		field_strength += *axis_ptr * neuron->membrane_potential;
	}

	if (face->neuron_count > 0) {
		face->neural_activity[axis_id] = field_strength
				/ (float) face->neuron_count;
	}

	pthread_mutex_unlock(&face->data_mutex);
}

/* ============================================================================
 * Thread Worker Function
 * =========================================================================== */

static void* thread_process_neurons(void *arg) {
	ThreadWork *work = (ThreadWork*) arg;
	struct timespec start, end;
	unsigned int i;
	float input_current;

	if (!work || !work->face)
		return NULL;
	if (!validate_face(work->face))
		return NULL;

	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
			&work->cpu_affinity);

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

	work->spike_count = 0;
	work->simd_ops = 0;
	work->running = 1;
	work->should_stop = 0;

	for (i = work->neuron_start;
			i < work->neuron_end && i < work->face->neuron_count; i++) {
		if (work->should_stop)
			break;

		DimensionalNeuron *neuron = &work->face->neurons[i];

		input_current = (neuron->position.x * work->axis_weights[0]
				+ neuron->position.y * work->axis_weights[1]
				+ neuron->position.z * work->axis_weights[2]
				+ neuron->position.d * work->axis_weights[3]
				+ neuron->position.a * work->axis_weights[4])
				* work->face->learning_rate;

		if (input_current > 1.0f)
			input_current = 1.0f;
		if (input_current < -1.0f)
			input_current = -1.0f;

		neural_update_spike(neuron, input_current);
		work->spike_count += neuron->spike_count;
		work->simd_ops += 5;
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

	work->completed = 1;
	work->running = 0;
	work->processing_time = (end.tv_sec - start.tv_sec)
			+ (end.tv_nsec - start.tv_nsec) / 1.0e9;

	return NULL;
}

/* ============================================================================
 * Thread Pool Initialization
 * =========================================================================== */

static int init_neural_thread_pool(AppState *state) {
	unsigned int i, j;
	int num_cores = get_nprocs();
	unsigned int neurons_per_thread;
	float axis_weight_sum = 0.0f;

	if (!state || !validate_face(&state->face))
		return -1;

	if (numa_available() >= 0) {
		state->numa_nodes = numa_num_configured_nodes();
	} else {
		state->numa_nodes = 1;
	}

	state->cpu_cores = num_cores;
	state->thread_count = (num_cores < MAX_THREADS) ? num_cores : MAX_THREADS;

	if (state->thread_count < 1)
		state->thread_count = 1;
	if (state->thread_count > MAX_THREADS)
		state->thread_count = MAX_THREADS;

	neurons_per_thread = state->face.neuron_count / state->thread_count;
	if (neurons_per_thread < 1)
		neurons_per_thread = 1;

	for (i = 0; i < AXIS_COUNT; i++) {
		axis_weight_sum += state->axis_activity[i];
	}

	if (axis_weight_sum < 0.001f) {
		axis_weight_sum = 1.0f;
		for (i = 0; i < AXIS_COUNT; i++) {
			state->axis_activity[i] = 1.0f / (float) AXIS_COUNT;
		}
	}

	for (i = 0; i < state->thread_count; i++) {
		ThreadWork *work = &state->threads[i];

		memset(work, 0, sizeof(ThreadWork));
		pthread_mutex_init(&work->work_mutex, NULL);

		work->thread_id = i;
		work->cpu_core = i % num_cores;
		work->face = &state->face;
		work->completed = 0;
		work->running = 0;
		work->should_stop = 0;

		if (numa_available() >= 0) {
			work->node_id = numa_node_of_cpu(work->cpu_core);
		} else {
			work->node_id = 0;
		}

		for (j = 0; j < AXIS_COUNT; j++) {
			work->axis_weights[j] = state->axis_activity[j] / axis_weight_sum;
			if (work->axis_weights[j] < 0.1f)
				work->axis_weights[j] = 0.1f;
			if (work->axis_weights[j] > 1.0f)
				work->axis_weights[j] = 1.0f;
		}

		CPU_ZERO(&work->cpu_affinity);
		CPU_SET(work->cpu_core, &work->cpu_affinity);

		work->neuron_start = i * neurons_per_thread;
		work->neuron_end =
				(i == state->thread_count - 1) ?
						state->face.neuron_count : (i + 1) * neurons_per_thread;

		if (pthread_create(&work->thread, NULL, thread_process_neurons, work)
				!= 0) {
			perror("Failed to create thread");
			pthread_mutex_destroy(&work->work_mutex);
			return -1;
		}
	}

	return 0;
}

/* ============================================================================
 * Stop Threads
 * =========================================================================== */

static void stop_threads(AppState *state) {
	unsigned int i;

	if (!state)
		return;

	for (i = 0; i < state->thread_count; i++) {
		pthread_mutex_lock(&state->threads[i].work_mutex);
		state->threads[i].should_stop = 1;
		pthread_mutex_unlock(&state->threads[i].work_mutex);
	}
}

/* ============================================================================
 * Wait for Threads
 * =========================================================================== */

static void wait_for_threads(AppState *state) {
	unsigned int i;
	int all_completed;
	unsigned int timeout = 0;
	const unsigned int max_timeout = 1000;

	if (!state)
		return;

	do {
		all_completed = 1;
		for (i = 0; i < state->thread_count; i++) {
			if (!state->threads[i].completed
					&& !state->threads[i].should_stop) {
				all_completed = 0;
				break;
			}
		}
		if (!all_completed) {
			sched_yield();
			timeout++;
			if (timeout > max_timeout) {
				break;
			}
		}
	} while (!all_completed && timeout < max_timeout);
}

/* ============================================================================
 * Join Threads
 * =========================================================================== */

static void join_threads(AppState *state) {
	unsigned int i;
	void *retval;

	if (!state)
		return;

	stop_threads(state);

	safe_sleep(50);

	for (i = 0; i < state->thread_count; i++) {
		if (state->threads[i].thread) {
			pthread_join(state->threads[i].thread, &retval);
			pthread_mutex_destroy(&state->threads[i].work_mutex);
			state->threads[i].thread = 0;
		}
	}
}

/* ============================================================================
 * Generate Neural Vector
 * =========================================================================== */

static NeuralVectorLine generate_neural_vector(unsigned int id,
		const DimensionalNeuron *pre, const DimensionalNeuron *post,
		unsigned int bundle_id) {
	NeuralVectorLine vec;
	unsigned char intensity;

	(void) id;

	memset(&vec, 0, sizeof(NeuralVectorLine));

	if (!pre || !post)
		return vec;

	vec.start = pre->position;
	vec.end = post->position;

	vec.direction.x = vec.end.x - vec.start.x;
	vec.direction.y = vec.end.y - vec.start.y;
	vec.direction.z = vec.end.z - vec.start.z;
	vec.direction.d = vec.end.d - vec.start.d;
	vec.direction.a = vec.end.a - vec.start.a;
	vec.direction.weight = vec.end.weight - vec.start.weight;

	vec.magnitude = neural_vector_length(vec.direction);
	if (vec.magnitude > 0.0f) {
		vec.direction = neural_vector_normalize(vec.direction);
	}

	vec.synaptic_strength = (pre->firing_threshold + post->firing_threshold)
			* 0.5f;
	if (vec.synaptic_strength > 1.0f)
		vec.synaptic_strength = 1.0f;

	vec.firing_rate = ((float) pre->spike_count + (float) post->spike_count)
			* 0.5f / 1000.0f;
	if (vec.firing_rate > 1.0f)
		vec.firing_rate = 1.0f;

	vec.presynaptic_id = pre->id;
	vec.postsynaptic_id = post->id;
	vec.axon_bundle = bundle_id;
	vec.plasticity = 0.1f;

	intensity = (unsigned char) (vec.synaptic_strength * 255.0f);
	if (intensity < 50)
		intensity = 50;

	switch (bundle_id % 5) {
	case 0:
		vec.color.r = 255;
		vec.color.g = intensity / 3;
		vec.color.b = intensity / 3;
		break;
	case 1:
		vec.color.r = intensity / 3;
		vec.color.g = 255;
		vec.color.b = intensity / 3;
		break;
	case 2:
		vec.color.r = intensity / 3;
		vec.color.g = intensity / 3;
		vec.color.b = 255;
		break;
	case 3:
		vec.color.r = 255;
		vec.color.g = intensity / 3;
		vec.color.b = 255;
		break;
	case 4:
		vec.color.r = intensity / 3;
		vec.color.g = 255;
		vec.color.b = 255;
		break;
	default:
		vec.color.r = vec.color.g = vec.color.b = 255;
		break;
	}
	vec.color.a = (unsigned char) (vec.firing_rate * 255.0f);
	if (vec.color.a < 50)
		vec.color.a = 50;

	return vec;
}

/* ============================================================================
 * Initialize Neural Face (embedded in AppState)
 * =========================================================================== */

static int init_neural_face(AppState *state) {
	unsigned int i, j, k;
	unsigned int active = 0;
	CADNeuralFace *face;

	if (!state)
		return -1;

	face = &state->face;
	memset(face, 0, sizeof(CADNeuralFace));

	if (pthread_mutex_init(&face->data_mutex, NULL) != 0) {
		fprintf(stderr, "Failed to initialize face mutex\n");
		return -1;
	}

	face->neuron_count = NEURON_COUNT;
	face->vector_count = VECTOR_COUNT;

	printf("  ├─ Allocating %d neurons... ", NEURON_COUNT);
	fflush(stdout);

	face->neurons = (DimensionalNeuron*) aligned_alloc(64,
	NEURON_COUNT * sizeof(DimensionalNeuron));

	if (!face->neurons) {
		fprintf(stderr, "Failed to allocate neurons\n");
		pthread_mutex_destroy(&face->data_mutex);
		return -1;
	}
	memset(face->neurons, 0, NEURON_COUNT * sizeof(DimensionalNeuron));
	printf("OK\n");

	printf("  ├─ Allocating %d vectors... ", VECTOR_COUNT);
	fflush(stdout);

	face->vectors = (NeuralVectorLine*) aligned_alloc(64,
	VECTOR_COUNT * sizeof(NeuralVectorLine));

	if (!face->vectors) {
		fprintf(stderr, "Failed to allocate vectors\n");
		free(face->neurons);
		face->neurons = NULL;
		pthread_mutex_destroy(&face->data_mutex);
		return -1;
	}
	memset(face->vectors, 0, VECTOR_COUNT * sizeof(NeuralVectorLine));
	printf("OK\n");

	printf("  ├─ Initializing neural network... ");
	fflush(stdout);

	for (i = 0; i < NEURAL_LAYERS; i++) {
		float layer_scale = (float) i / (float) NEURAL_LAYERS;
		unsigned int layer_start = (i * face->neuron_count) / NEURAL_LAYERS;
		unsigned int layer_end = ((i + 1) * face->neuron_count) / NEURAL_LAYERS;

		for (j = layer_start; j < layer_end; j++) {
			DimensionalNeuron *neuron = &face->neurons[j];

			float u = (float) (j % 512) / 512.0f;
			float v = (float) ((j / 512) % 512) / 512.0f;
			float w = (float) (j / (512 * 512))
					/ (float) (face->neuron_count / (512 * 512));

			float theta = u * 2.0f * 3.14159265359f;
			float phi = (v - 0.5f) * 3.14159265359f;
			float psi = w * 2.0f * 3.14159265359f;

			neuron->position.x = (float) (sin(phi) * cos(theta)
					* (2.0f + 0.5f * sin(psi * 2.0f)));
			neuron->position.y = (float) (sin(phi) * sin(theta) * 1.8f
					* (1.0f + 0.3f * cos(psi)));
			neuron->position.z = (float) (cos(phi) * 2.4f
					* (1.0f + 0.4f * sin(psi * 1.5f)));
			neuron->position.d = (float) (sin(psi) * 1.0f);
			neuron->position.a = (float) (cos(psi) * 1.0f);

			neuron->position.weight = 1.0f;
			neuron->learning_rate = 0.01f * (1.0f - layer_scale * 0.5f);
			neuron->position.learning_rate = neuron->learning_rate;
			neuron->position.spike_rate = 0.0f;

			neuron->membrane_potential = (float) rand() / (float) RAND_MAX
					* 0.1f;
			neuron->firing_threshold = 0.5f + layer_scale * 0.3f;
			neuron->refractory_period = 0.0f;
			neuron->adaptation_rate = 0.1f + layer_scale * 0.2f;
			neuron->layer_id = i;
			neuron->id = j;
			neuron->spike_count = 0;
			neuron->history_index = 0;
			neuron->connection_count = 0;

			for (k = 0; k < AXIS_COUNT; k++) {
				neuron->synaptic_weight[k] = 0.5f
						+ 0.5f * (float) rand() / (float) RAND_MAX;
			}

			for (k = 0; k < MAX_SPIKE_HISTORY; k++) {
				neuron->firing_history[k] = 0;
			}

			neuron->color.r = (unsigned char) (255
					* (0.3f + 0.7f * (float) i / NEURAL_LAYERS));
			neuron->color.g = (unsigned char) (255
					* (0.2f + 0.8f * neuron->firing_threshold));
			neuron->color.b = (unsigned char) (255
					* (0.1f + 0.9f * neuron->adaptation_rate));
			neuron->color.a = 200;
		}
	}
	printf("OK\n");

	printf("  ├─ Generating vector field... ");
	fflush(stdout);

	for (i = 0; i < face->vector_count; i++) {
		unsigned int pre_id = (i * 2654435761u) % face->neuron_count;
		unsigned int post_id = (pre_id + 1 + (i % 100)) % face->neuron_count;
		unsigned int bundle_id = i % AXIS_COUNT;

		face->vectors[i] = generate_neural_vector(i, &face->neurons[pre_id],
				&face->neurons[post_id], bundle_id);

		if (face->vectors[i].magnitude > 0.1f) {
			active++;
		}
	}
	printf("OK\n");

	face->active_connections = active;
	face->learning_rate = 0.01f;
	face->initialized = 1;

	return 0;
}

/* ============================================================================
 * Calculate Face Bounds
 * =========================================================================== */

static void calculate_face_bounds(AppState *state) {
	unsigned int i;
	float min_x, max_x, min_y, max_y, min_z, max_z;
	CADNeuralFace *face;

	if (!state)
		return;

	face = &state->face;
	if (!face->initialized)
		return;
	if (!face->neurons || face->neuron_count == 0)
		return;

	pthread_mutex_lock(&face->data_mutex);

	min_x = max_x = face->neurons[0].position.x;
	min_y = max_y = face->neurons[0].position.y;
	min_z = max_z = face->neurons[0].position.z;

	for (i = 1; i < face->neuron_count; i++) {
		DimensionalNeuron *n = &face->neurons[i];

		if (n->position.x < min_x)
			min_x = n->position.x;
		if (n->position.x > max_x)
			max_x = n->position.x;
		if (n->position.y < min_y)
			min_y = n->position.y;
		if (n->position.y > max_y)
			max_y = n->position.y;
		if (n->position.z < min_z)
			min_z = n->position.z;
		if (n->position.z > max_z)
			max_z = n->position.z;
	}

	face->center[0] = (min_x + max_x) * 0.5f;
	face->center[1] = (min_y + max_y) * 0.5f;
	face->center[2] = (min_z + max_z) * 0.5f;

	face->radius = (float) sqrt(
			(double) ((max_x - min_x) * (max_x - min_x)
					+ (max_y - min_y) * (max_y - min_y)
					+ (max_z - min_z) * (max_z - min_z))) * 0.5f;

	if (face->radius < 1.0f)
		face->radius = 1.0f;

	pthread_mutex_unlock(&face->data_mutex);
}

/* ============================================================================
 * Free Neural Face (embedded in AppState)
 * =========================================================================== */

static void free_neural_face(AppState *state) {
	CADNeuralFace *face;

	if (!state)
		return;

	face = &state->face;

	pthread_mutex_lock(&face->data_mutex);

	face->initialized = 0;

	if (face->neurons) {
		free(face->neurons);
		face->neurons = NULL;
	}

	if (face->vectors) {
		free(face->vectors);
		face->vectors = NULL;
	}

	pthread_mutex_unlock(&face->data_mutex);
	pthread_mutex_destroy(&face->data_mutex);
}

/* ============================================================================
 * Validate OpenGL State
 * =========================================================================== */

static int validate_gl_state(GLContext *gl) {
	if (!gl)
		return 0;
	if (!gl->initialized)
		return 0;
	if (!gl->window)
		return 0;
	if (!gl->gl_context)
		return 0;

	if (SDL_GL_MakeCurrent(gl->window, gl->gl_context) != 0) {
		return 0;
	}

	return 1;
}

/* ============================================================================
 * OpenGL Functions
 * =========================================================================== */

static int init_neural_opengl(GLContext *gl) {
	int i;

	if (!gl)
		return -1;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
		return -1;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

	gl->window = SDL_CreateWindow(
	PROGRAM_NAME " " PROGRAM_VERSION,
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED,
	WINDOW_WIDTH,
	WINDOW_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	if (!gl->window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit();
		return -1;
	}

	gl->gl_context = SDL_GL_CreateContext(gl->window);
	if (!gl->gl_context) {
		fprintf(stderr, "OpenGL context creation failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		return -1;
	}

	if (SDL_GL_SetSwapInterval(1) < 0) {
		fprintf(stderr, "Warning: Unable to set VSync: %s\n", SDL_GetError());
	}

	gl->clear_color[0] = 0.05f;
	gl->clear_color[1] = 0.05f;
	gl->clear_color[2] = 0.1f;
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

	gl->window_width = WINDOW_WIDTH;
	gl->window_height = WINDOW_HEIGHT;
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) WINDOW_WIDTH / (double) WINDOW_HEIGHT, 0.1,
			100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	while (glGetError() != GL_NO_ERROR) {
	}

	gl->neuron_list = glGenLists(1);
	gl->vector_list = glGenLists(1);
	for (i = 0; i < AXIS_COUNT; i++) {
		gl->axis_display_list[i] = glGenLists(1);
	}

	gl->initialized = 1;
	gl->render_ready = 1;

	return 0;
}

static void build_neural_display_lists(AppState *state) {
	unsigned int i, j;
	CADNeuralFace *face;

	if (!state)
		return;
	if (!state->gl.initialized)
		return;
	if (!validate_face(&state->face))
		return;
	if (!validate_gl_state(&state->gl))
		return;

	face = &state->face;

	glNewList(state->gl.neuron_list, GL_COMPILE);
	glPointSize(POINT_SIZE);
	glBegin(GL_POINTS);

	for (i = 0; i < face->neuron_count; i++) {
		DimensionalNeuron *neuron = &face->neurons[i];
		float activity = 0.5f + 0.5f * ((float) neuron->spike_count / 100.0f);
		if (activity > 1.0f)
			activity = 1.0f;

		glColor4f((float) neuron->color.r / 255.0f * activity,
				(float) neuron->color.g / 255.0f * activity,
				(float) neuron->color.b / 255.0f * activity, 0.8f);

		glVertex3f(neuron->position.x, neuron->position.y, neuron->position.z);
	}
	glEnd();
	glEndList();

	glNewList(state->gl.vector_list, GL_COMPILE);
	glLineWidth(LINE_WIDTH);
	glBegin(GL_LINES);

	for (i = 0; i < face->vector_count; i++) {
		NeuralVectorLine *vec = &face->vectors[i];
		float alpha = vec->firing_rate * 0.5f;
		if (alpha < 0.1f)
			alpha = 0.1f;
		if (alpha > 1.0f)
			alpha = 1.0f;

		glColor4f((float) vec->color.r / 255.0f, (float) vec->color.g / 255.0f,
				(float) vec->color.b / 255.0f, alpha);

		glVertex3f(vec->start.x, vec->start.y, vec->start.z);
		glVertex3f(vec->end.x, vec->end.y, vec->end.z);
	}
	glEnd();
	glEndList();

	for (j = 0; j < AXIS_COUNT; j++) {
		glNewList(state->gl.axis_display_list[j], GL_COMPILE);
		glLineWidth(LINE_WIDTH * 1.5f);
		glBegin(GL_LINES);

		for (i = 0; i < face->vector_count; i++) {
			NeuralVectorLine *vec = &face->vectors[i];
			if (vec->axon_bundle == j) {
				float axis_val;
				switch (j) {
				case 0:
					axis_val = (float) fabs(vec->direction.x);
					break;
				case 1:
					axis_val = (float) fabs(vec->direction.y);
					break;
				case 2:
					axis_val = (float) fabs(vec->direction.z);
					break;
				case 3:
					axis_val = (float) fabs(vec->direction.d);
					break;
				case 4:
					axis_val = (float) fabs(vec->direction.a);
					break;
				default:
					axis_val = 0.0f;
					break;
				}

				if (axis_val > 0.1f && axis_val <= 1.0f) {
					glColor4f((j == 0) ? axis_val : axis_val * 0.3f,
							(j == 1) ? axis_val : axis_val * 0.3f,
							(j == 2) ? axis_val : axis_val * 0.3f, 0.3f);
					glVertex3f(vec->start.x, vec->start.y, vec->start.z);
					glVertex3f(vec->end.x, vec->end.y, vec->end.z);
				}
			}
		}
		glEnd();
		glEndList();
	}

	check_gl_error("Building display lists");
}

static void render_neural_cad_face(AppState *state) {
	float cam_x, cam_y, cam_z;
	int i;
	CADNeuralFace *face;

	if (!state)
		return;
	if (state->shutting_down)
		return;
	if (!state->gl.initialized)
		return;
	if (!state->gl.render_ready)
		return;
	if (!validate_face(&state->face))
		return;

	if (pthread_mutex_trylock(&state->render_mutex) != 0) {
		return;
	}

	if (!validate_gl_state(&state->gl)) {
		pthread_mutex_unlock(&state->render_mutex);
		return;
	}

	face = &state->face;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,
			(double) state->gl.window_width / (double) state->gl.window_height,
			0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	pthread_mutex_lock(&state->state_mutex);
	cam_x = (float) (sin(state->camera_angle) * state->camera_distance);
	cam_y = state->camera_height;
	cam_z = (float) (cos(state->camera_angle) * state->camera_distance);
	pthread_mutex_unlock(&state->state_mutex);

	gluLookAt(cam_x, cam_y, cam_z, face->center[0], face->center[1],
			face->center[2], 0.0f, 1.0f, 0.0f);

	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor4f(0.2f, 0.2f, 0.2f, 0.3f);
	for (i = -10; i <= 10; i++) {
		float pos = i * 0.5f;
		glVertex3f(pos, -2.0f, -5.0f);
		glVertex3f(pos, -2.0f, 5.0f);
		glVertex3f(-5.0f, -2.0f, pos);
		glVertex3f(5.0f, -2.0f, pos);
	}
	glEnd();

	glCallList(state->gl.neuron_list);
	glCallList(state->gl.vector_list);

	glEnable(GL_BLEND);
	for (i = 0; i < AXIS_COUNT; i++) {
		if (face->neural_activity[i] > 0.1f) {
			glCallList(state->gl.axis_display_list[i]);
		}
	}
	glDisable(GL_BLEND);

	check_gl_error("Rendering");

	SDL_GL_SwapWindow(state->gl.window);

	pthread_mutex_unlock(&state->render_mutex);
}

static void check_gl_error(const char *operation) {
	GLenum error;
	int error_count = 0;

	while ((error = glGetError()) != GL_NO_ERROR && error_count < 10) {
		fprintf(stderr, "OpenGL error after %s: 0x%x\n", operation, error);
		error_count++;
	}
}

static void resize_window(GLContext *gl, int width, int height) {
	if (!gl || !gl->initialized)
		return;
	if (width <= 0 || height <= 0)
		return;

	gl->window_width = width;
	gl->window_height = height;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / (double) height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
}

static void cleanup_opengl(GLContext *gl) {
	int i;

	if (!gl)
		return;

	gl->render_ready = 0;

	if (gl->initialized) {
		if (gl->neuron_list)
			glDeleteLists(gl->neuron_list, 1);
		if (gl->vector_list)
			glDeleteLists(gl->vector_list, 1);
		for (i = 0; i < AXIS_COUNT; i++) {
			if (gl->axis_display_list[i])
				glDeleteLists(gl->axis_display_list[i], 1);
		}

		if (gl->gl_context) {
			SDL_GL_DeleteContext(gl->gl_context);
			gl->gl_context = NULL;
		}

		if (gl->window) {
			SDL_DestroyWindow(gl->window);
			gl->window = NULL;
		}

		gl->initialized = 0;
	}
}

/* ============================================================================
 * OpenCL Functions
 * =========================================================================== */

static int init_neural_opencl(CLContext *cl) {
	cl_uint platform_count;
	cl_uint device_count;
	cl_int err;

	if (!cl)
		return -1;

	cl->initialized = 0;

	err = clGetPlatformIDs(0, NULL, &platform_count);
	if (err != CL_SUCCESS || platform_count == 0) {
		return -1;
	}

	err = clGetPlatformIDs(1, &cl->platform, NULL);
	if (err != CL_SUCCESS)
		return -1;

	err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 0, NULL,
			&device_count);
	if (err != CL_SUCCESS || device_count == 0) {
		return -1;
	}

	err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 1, &cl->device,
			NULL);
	if (err != CL_SUCCESS)
		return -1;

	cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
	if (err != CL_SUCCESS)
		return -1;

#ifdef CL_VERSION_2_0
	cl_queue_properties properties[] = { 0 };
	cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device,
			properties, &err);
#else
    cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
#endif

	if (err != CL_SUCCESS) {
		clReleaseContext(cl->context);
		return -1;
	}

	cl->global_work_size = NEURON_COUNT;
	cl->local_work_size = 256;
	cl->initialized = 1;

	return 0;
}

static void cleanup_opencl(CLContext *cl) {
	if (!cl)
		return;

	if (cl->queue) {
		clReleaseCommandQueue(cl->queue);
		cl->queue = NULL;
	}

	if (cl->context) {
		clReleaseContext(cl->context);
		cl->context = NULL;
	}

	cl->initialized = 0;
}

/* ============================================================================
 * OpenAL Functions
 * =========================================================================== */

static int init_openal(AppState *state) {
	if (!state)
		return -1;

	ALCdevice *device = alcOpenDevice(NULL);
	if (!device) {
		return -1;
	}

	ALCcontext *context = alcCreateContext(device, NULL);
	if (!context) {
		alcCloseDevice(device);
		return -1;
	}

	if (alcMakeContextCurrent(context) == ALC_FALSE) {
		alcDestroyContext(context);
		alcCloseDevice(device);
		return -1;
	}

	alListener3f(AL_POSITION, 0.0f, 0.0f, 5.0f);

	state->al_device = device;
	state->al_context = context;

	return 0;
}

static void cleanup_openal(AppState *state) {
	if (!state)
		return;

	if (state->al_context) {
		alcMakeContextCurrent(NULL);
		alcDestroyContext(state->al_context);
		state->al_context = NULL;
	}

	if (state->al_device) {
		alcCloseDevice(state->al_device);
		state->al_device = NULL;
	}
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
	struct timespec req, rem;
	req.tv_sec = ms / 1000;
	req.tv_nsec = (ms % 1000) * 1000000;
	nanosleep(&req, &rem);
}

static void print_system_info(AppState *state) {
	printf(
			"\n╔══════════════════════════════════════════════════════════════════════╗\n");
	printf(
			"║         MIMIX CAD Face - Neural Dimensional System v%s            ║\n",
			PROGRAM_VERSION);
	printf(
			"╚══════════════════════════════════════════════════════════════════════╝\n\n");

	printf("System Configuration:\n");
	printf("  ├─ CPU Cores: %d\n", state->cpu_cores);
	printf("  ├─ NUMA Nodes: %d\n", state->numa_nodes);
	printf("  ├─ Memory Alignment: %d bytes\n", SIMD_ALIGNMENT);
	printf("  └─ Cache Line: %d bytes\n", CACHE_LINE_SIZE);

	printf("\nNeural Network:\n");
	printf("  ├─ Processing Axes: %d\n", AXIS_COUNT);
	printf("  ├─ Neural Vectors: %d\n", VECTOR_COUNT);
	printf("  ├─ Dimensional Neurons: %d\n", NEURON_COUNT);
	printf("  ├─ Neural Layers: %d\n", NEURAL_LAYERS);
	printf("  └─ Synapse Density: %.2f\n", SYNAPSE_DENSITY);
}

static void handle_events(AppState *state) {
	SDL_Event event;
	unsigned int i;

	if (!state)
		return;
	if (state->shutting_down)
		return;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			state->running = 0;
			break;

		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE:
				state->running = 0;
				break;

			case SDLK_SPACE:
				pthread_mutex_lock(&state->state_mutex);
				state->paused = !state->paused;
				pthread_mutex_unlock(&state->state_mutex);
				printf("%s\n", state->paused ? "PAUSED" : "RUNNING");
				break;

			case SDLK_1:
			case SDLK_2:
			case SDLK_3:
			case SDLK_4:
			case SDLK_5:
				i = event.key.keysym.sym - SDLK_1;
				if (i < AXIS_COUNT) {
					pthread_mutex_lock(&state->state_mutex);
					state->axis_activity[i] =
							(state->axis_activity[i] > 0.5f) ? 0.1f : 1.0f;
					pthread_mutex_unlock(&state->state_mutex);
				}
				break;

			case SDLK_r:
				pthread_mutex_lock(&state->state_mutex);
				state->camera_angle = 0.0f;
				state->camera_distance = 8.0f;
				state->camera_height = 2.0f;
				state->rotation_speed = 0.02f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_UP:
				pthread_mutex_lock(&state->state_mutex);
				state->camera_distance -= 0.5f;
				if (state->camera_distance < 3.0f)
					state->camera_distance = 3.0f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_DOWN:
				pthread_mutex_lock(&state->state_mutex);
				state->camera_distance += 0.5f;
				if (state->camera_distance > 20.0f)
					state->camera_distance = 20.0f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_LEFT:
				pthread_mutex_lock(&state->state_mutex);
				state->rotation_speed -= 0.005f;
				if (state->rotation_speed < 0.0f)
					state->rotation_speed = 0.0f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_RIGHT:
				pthread_mutex_lock(&state->state_mutex);
				state->rotation_speed += 0.005f;
				if (state->rotation_speed > 0.05f)
					state->rotation_speed = 0.05f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_PAGEUP:
				pthread_mutex_lock(&state->state_mutex);
				state->camera_height += 0.5f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			case SDLK_PAGEDOWN:
				pthread_mutex_lock(&state->state_mutex);
				state->camera_height -= 0.5f;
				pthread_mutex_unlock(&state->state_mutex);
				break;

			default:
				break;
			}
			break;

		case SDL_WINDOWEVENT:
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				resize_window(&state->gl, event.window.data1,
						event.window.data2);
			}
			break;
		}
	}
}

static volatile int g_running = 1;

static void signal_handler(int sig) {
	(void) sig;
	g_running = 0;
}

/* ============================================================================
 * Cleanup All Resources
 * =========================================================================== */

static void cleanup_all(AppState *state) {
	if (!state)
		return;

	pthread_mutex_lock(&state->cleanup_mutex);

	if (state->shutting_down) {
		pthread_mutex_unlock(&state->cleanup_mutex);
		return;
	}

	state->shutting_down = 1;
	state->running = 0;
	state->gl.render_ready = 0;

	pthread_mutex_unlock(&state->cleanup_mutex);

	join_threads(state);

	cleanup_opengl(&state->gl);
	cleanup_openal(state);
	cleanup_opencl(&state->cl);

	if (state->face.initialized) {
		free_neural_face(state);
	}
}

/* ============================================================================
 * Main Function - Fixed memory management
 * =========================================================================== */

int main(int argc, char **argv) {
	AppState state;
	unsigned int current_time;
	unsigned int i;

	(void) argc;
	(void) argv;

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	srand((unsigned int) time(NULL));

	memset(&state, 0, sizeof(AppState));

	if (pthread_mutex_init(&state.render_mutex, NULL) != 0) {
		fprintf(stderr, "Failed to initialize render mutex\n");
		return EXIT_FAILURE;
	}

	if (pthread_mutex_init(&state.state_mutex, NULL) != 0) {
		fprintf(stderr, "Failed to initialize state mutex\n");
		pthread_mutex_destroy(&state.render_mutex);
		return EXIT_FAILURE;
	}

	if (pthread_mutex_init(&state.cleanup_mutex, NULL) != 0) {
		fprintf(stderr, "Failed to initialize cleanup mutex\n");
		pthread_mutex_destroy(&state.render_mutex);
		pthread_mutex_destroy(&state.state_mutex);
		return EXIT_FAILURE;
	}

	for (i = 0; i < AXIS_COUNT; i++) {
		state.axis_activity[i] = 1.0f / (float) AXIS_COUNT;
	}

	state.camera_angle = 0.0f;
	state.camera_distance = 8.0f;
	state.camera_height = 2.0f;
	state.rotation_speed = 0.02f;
	state.running = 1;
	state.paused = 0;
	state.shutting_down = 0;

	state.cpu_cores = get_nprocs();
	if (numa_available() >= 0) {
		state.numa_nodes = numa_num_configured_nodes();
	} else {
		state.numa_nodes = 1;
	}

	print_system_info(&state);

	printf("\nGenerating Neural CAD face...\n");

	if (init_neural_face(&state) != 0) {
		fprintf(stderr, "Failed to initialize neural face\n");
		pthread_mutex_destroy(&state.render_mutex);
		pthread_mutex_destroy(&state.state_mutex);
		pthread_mutex_destroy(&state.cleanup_mutex);
		return EXIT_FAILURE;
	}

	calculate_face_bounds(&state);

	printf("\n  ├─ Dimensional Neurons: %d\n", state.face.neuron_count);
	printf("  ├─ Neural Vectors: %d\n", state.face.vector_count);
	printf("  ├─ Active Connections: %d\n", state.face.active_connections);
	printf("  └─ Face Center: (%.2f, %.2f, %.2f)\n\n", state.face.center[0],
			state.face.center[1], state.face.center[2]);

	printf("Initializing thread pool... ");
	fflush(stdout);
	if (init_neural_thread_pool(&state) != 0) {
		fprintf(stderr, "Failed to initialize neural thread pool\n");
		free_neural_face(&state);
		pthread_mutex_destroy(&state.render_mutex);
		pthread_mutex_destroy(&state.state_mutex);
		pthread_mutex_destroy(&state.cleanup_mutex);
		return EXIT_FAILURE;
	}
	printf("OK (%d threads)\n", state.thread_count);

	printf("Initializing OpenCL... ");
	fflush(stdout);
	if (init_neural_opencl(&state.cl) != 0) {
		printf("Skipped\n");
	} else {
		printf("OK\n");
	}

	printf("Initializing OpenGL... ");
	fflush(stdout);
	if (init_neural_opengl(&state.gl) != 0) {
		fprintf(stderr, "Failed to initialize OpenGL\n");
		join_threads(&state);
		free_neural_face(&state);
		pthread_mutex_destroy(&state.render_mutex);
		pthread_mutex_destroy(&state.state_mutex);
		pthread_mutex_destroy(&state.cleanup_mutex);
		return EXIT_FAILURE;
	}
	printf("OK\n");

	printf("Building display lists... ");
	fflush(stdout);
	build_neural_display_lists(&state);
	printf("OK\n");

	printf("Initializing OpenAL... ");
	fflush(stdout);
	if (init_openal(&state) != 0) {
		printf("Skipped\n");
	} else {
		printf("OK\n");
	}

	printf(
			"\n╔══════════════════════════════════════════════════════════════════════╗\n");
	printf(
			"║                    Controls                                          ║\n");
	printf(
			"║  [ESC] Exit    [SPACE] Pause    [R] Reset Camera                     ║\n");
	printf(
			"║  [1-5] Toggle Axis    [UP/DOWN] Zoom    [LEFT/RIGHT] Speed          ║\n");
	printf(
			"║  [PGUP/PGDN] Height                                                   ║\n");
	printf(
			"╚══════════════════════════════════════════════════════════════════════╝\n\n");

	state.last_frame_time = get_time_ms();
	state.frame_count = 0;
	state.fps = 0.0f;

	while (state.running && g_running && !state.shutting_down) {
		current_time = get_time_ms();

		handle_events(&state);

		if (!state.paused && !state.shutting_down
				&& validate_face(&state.face)) {
			wait_for_threads(&state);

			for (i = 0; i < AXIS_COUNT; i++) {
				process_5axis_neural_field(&state.face, i);
			}

			for (i = 0; i < state.thread_count; i++) {
				state.threads[i].completed = 0;
			}
		}

		render_neural_cad_face(&state);

		state.frame_count++;

		if (current_time - state.last_frame_time >= 1000) {
			if (current_time > state.last_frame_time) {
				state.fps = (double) state.frame_count * 1000.0
						/ (double) (current_time - state.last_frame_time);
			}

			pthread_mutex_lock(&state.state_mutex);
			state.total_spikes = 0;
			state.total_simd_ops = 0;
			for (i = 0; i < state.thread_count; i++) {
				state.total_spikes += state.threads[i].spike_count;
				state.total_simd_ops += state.threads[i].simd_ops;
			}
			pthread_mutex_unlock(&state.state_mutex);

			printf("\r║ FPS: %6.2f ║ Spikes: %8lu ║ %s   ", state.fps,
					state.total_spikes, state.paused ? "PAUSED" : "RUNNING");
			fflush(stdout);

			state.frame_count = 0;
			state.last_frame_time = current_time;
		}

		if (!state.paused) {
			pthread_mutex_lock(&state.state_mutex);
			state.camera_angle += state.rotation_speed;
			pthread_mutex_unlock(&state.state_mutex);
		}

		safe_sleep(FRAME_TIME_MS);
	}

	printf("\n\nShutting down...\n");

	cleanup_all(&state);

	pthread_mutex_destroy(&state.render_mutex);
	pthread_mutex_destroy(&state.state_mutex);
	pthread_mutex_destroy(&state.cleanup_mutex);

	SDL_Quit();

	printf("\nNeural system shutdown complete. Goodbye.\n");

	return EXIT_SUCCESS;
}
