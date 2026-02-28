/**
 * @file mimix/src/main.c
 * @version 5.0.4
 * @license GNU 3
 * @author Evolutia Technologies
 *
 * @title MIMIX CAD Face - Neural Dimensional System
 * @description C90 compliant - Fixed compilation errors
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

/* OpenGL - Core Profile for stability */
#include <GL/gl.h>
#include <GL/glu.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>

/* SIMD Headers */
#include <immintrin.h>
#include <x86intrin.h>

/* ============================================================================
 * Compile-time Configuration
 * =========================================================================== */

#define PROGRAM_NAME         "MIMIX CAD Face"
#define PROGRAM_VERSION      "5.0.4"

/* Window Configuration - Safe defaults */
#define WINDOW_WIDTH         1024
#define WINDOW_HEIGHT        768
#define WINDOW_BPP           32

/* Neural Network Configuration */
#define AXIS_COUNT           5
#define VECTOR_COUNT         65536
#define NEURON_COUNT         262144
#define NEURAL_LAYERS        12
#define SYNAPSE_DENSITY      0.15f
#define MAX_SPIKE_HISTORY    64

/* Thread Configuration */
#define MAX_THREADS          4  /* Conservative for stability */
#define FRAME_TIME_MS        33 /* ~30 FPS */

/* Rendering Parameters */
#define POINT_SIZE           2.0f
#define LINE_WIDTH           1.0f

/* Memory alignment */
#define CACHE_LINE_SIZE      64
#define SIMD_ALIGNMENT       32
#define ALIGNED(align) __attribute__((aligned(align)))
#define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#define SIMD_ALIGNED __attribute__((aligned(SIMD_ALIGNMENT)))

/* ============================================================================
 * Core Data Structures - Simplified for stability
 * =========================================================================== */

typedef struct SIMD_ALIGNED NeuralVector5D {
	float x, y, z, d, a;
	float weight;
} NeuralVector5D;

typedef struct ColorRGB {
	unsigned char r, g, b, a;
} ColorRGB;

typedef struct CACHE_ALIGNED DimensionalNeuron {
	NeuralVector5D pos;
	float potential;
	float threshold;
	float refractory;
	float adaptation;
	float weights[AXIS_COUNT];
	unsigned int spikes;
	ColorRGB color;
	unsigned int id;
} DimensionalNeuron;

typedef struct CACHE_ALIGNED NeuralVectorLine {
	NeuralVector5D start, end;
	float strength;
	unsigned int bundle;
	ColorRGB color;
} NeuralVectorLine;

typedef struct CACHE_ALIGNED CADNeuralFace {
	DimensionalNeuron *neurons;
	NeuralVectorLine *vectors;
	unsigned int neuron_count;
	unsigned int vector_count;
	float center[3];
	float radius;
	pthread_mutex_t mutex;
	int initialized;
} CADNeuralFace;

typedef struct CACHE_ALIGNED AppState {
	CADNeuralFace face;
	pthread_t threads[MAX_THREADS];
	unsigned int thread_count;
	volatile int running;
	volatile int paused;
	float camera_angle;
	float camera_dist;
	float camera_height;
	double fps;
	struct timespec last_frame;
	int frame_count;
} AppState;

/* ============================================================================
 * OpenGL State - Isolated for safety
 * =========================================================================== */

typedef struct {
	SDL_Window *window;
	SDL_GLContext context;
	int width;
	int height;
	int initialized;
	GLuint neuron_list;
	GLuint vector_list;
} GLState;

static GLState g_gl = { 0 };

/* ============================================================================
 * Function Prototypes
 * =========================================================================== */

/* Core neural functions */
static int init_neural_face(AppState *state);
static void free_neural_face(AppState *state);
static void update_neurons(AppState *state);
static void* thread_worker(void *arg);

/* OpenGL functions - Isolated and safe */
static int init_opengl(void);
static void shutdown_opengl(void);
static void build_display_lists(AppState *state);
static void render_frame(AppState *state);
static void check_gl_error(const char *location);

/* Utility */
static unsigned int get_time_ms(void);
static void handle_events(AppState *state);
static void signal_handler(int sig);
static void safe_sleep(unsigned int ms);

/* ============================================================================
 * OpenGL Implementation - Carefully crafted for stability
 * =========================================================================== */

static int init_opengl(void) {
	/* Initialize SDL Video */
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return 0;
	}

	/* Set OpenGL attributes - Use older version for compatibility */
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16); /* Smaller depth buffer */
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0); /* Disable stencil */
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0); /* Disable MSAA initially */

	/* Create window */
	g_gl.window = SDL_CreateWindow(
	PROGRAM_NAME,
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED,
	WINDOW_WIDTH,
	WINDOW_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	if (!g_gl.window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit();
		return 0;
	}

	/* Create OpenGL context */
	g_gl.context = SDL_GL_CreateContext(g_gl.window);
	if (!g_gl.context) {
		fprintf(stderr, "Context creation failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(g_gl.window);
		SDL_Quit();
		return 0;
	}

	/* Try to enable vsync (optional) */
	SDL_GL_SetSwapInterval(1);

	/* Basic OpenGL setup */
	glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPointSize(POINT_SIZE);
	glLineWidth(LINE_WIDTH);

	/* Set viewport */
	g_gl.width = WINDOW_WIDTH;
	g_gl.height = WINDOW_HEIGHT;
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	/* Set projection */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) WINDOW_WIDTH / (double) WINDOW_HEIGHT, 0.1,
			100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Clear any errors */
	check_gl_error("init");

	g_gl.initialized = 1;
	printf("OpenGL initialized successfully\n");

	return 1;
}

static void shutdown_opengl(void) {
	if (g_gl.initialized) {
		/* Delete display lists */
		if (g_gl.neuron_list)
			glDeleteLists(g_gl.neuron_list, 1);
		if (g_gl.vector_list)
			glDeleteLists(g_gl.vector_list, 1);

		/* Destroy context and window */
		if (g_gl.context) {
			SDL_GL_DeleteContext(g_gl.context);
			g_gl.context = NULL;
		}
		if (g_gl.window) {
			SDL_DestroyWindow(g_gl.window);
			g_gl.window = NULL;
		}
		g_gl.initialized = 0;
	}
}

static void build_display_lists(AppState *state) {
	unsigned int i;
	CADNeuralFace *face = &state->face;

	if (!g_gl.initialized)
		return;
	if (!face->initialized)
		return;

	/* Make context current */
	SDL_GL_MakeCurrent(g_gl.window, g_gl.context);

	/* Delete old lists if they exist */
	if (g_gl.neuron_list)
		glDeleteLists(g_gl.neuron_list, 1);
	if (g_gl.vector_list)
		glDeleteLists(g_gl.vector_list, 1);

	/* Create neuron point list */
	g_gl.neuron_list = glGenLists(1);
	glNewList(g_gl.neuron_list, GL_COMPILE);
	glBegin(GL_POINTS);

	for (i = 0; i < face->neuron_count; i++) {
		DimensionalNeuron *n = &face->neurons[i];
		float intensity = 0.5f + 0.5f * (n->spikes / 100.0f);
		if (intensity > 1.0f)
			intensity = 1.0f;

		glColor4f(n->color.r / 255.0f * intensity,
				n->color.g / 255.0f * intensity,
				n->color.b / 255.0f * intensity, 0.8f);
		glVertex3f(n->pos.x, n->pos.y, n->pos.z);
	}
	glEnd();
	glEndList();

	/* Create vector line list */
	g_gl.vector_list = glGenLists(1);
	glNewList(g_gl.vector_list, GL_COMPILE);
	glBegin(GL_LINES);

	for (i = 0; i < face->vector_count; i++) {
		NeuralVectorLine *v = &face->vectors[i];
		float alpha = v->strength * 0.3f;
		if (alpha < 0.1f)
			alpha = 0.1f;

		glColor4f(v->color.r / 255.0f, v->color.g / 255.0f, v->color.b / 255.0f,
				alpha);
		glVertex3f(v->start.x, v->start.y, v->start.z);
		glVertex3f(v->end.x, v->end.y, v->end.z);
	}
	glEnd();
	glEndList();

	check_gl_error("build_lists");
}

static void render_frame(AppState *state) {
	float cam_x, cam_y, cam_z;
	int i; /* Declared at top for C90 compliance */

	if (!g_gl.initialized)
		return;
	if (!state->face.initialized)
		return;

	/* Make context current */
	SDL_GL_MakeCurrent(g_gl.window, g_gl.context);

	/* Clear buffers */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Reset matrices */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) g_gl.width / (double) g_gl.height, 0.1,
			100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Set camera */
	cam_x = (float) (sin(state->camera_angle) * state->camera_dist);
	cam_y = state->camera_height;
	cam_z = (float) (cos(state->camera_angle) * state->camera_dist);

	gluLookAt(cam_x, cam_y, cam_z, state->face.center[0], state->face.center[1],
			state->face.center[2], 0.0f, 1.0f, 0.0f);

	/* Draw simple reference grid */
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor4f(0.2f, 0.2f, 0.2f, 0.3f);
	for (i = -5; i <= 5; i++) { /* i declared at top */
		float pos = i * 0.5f;
		glVertex3f(pos, -1.0f, -2.5f);
		glVertex3f(pos, -1.0f, 2.5f);
		glVertex3f(-2.5f, -1.0f, pos);
		glVertex3f(2.5f, -1.0f, pos);
	}
	glEnd();

	/* Draw neurons and vectors */
	glCallList(g_gl.neuron_list);
	glCallList(g_gl.vector_list);

	check_gl_error("render");

	/* Swap buffers */
	SDL_GL_SwapWindow(g_gl.window);
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
 * Neural Network Implementation
 * =========================================================================== */

static int init_neural_face(AppState *state) {
	unsigned int i, j; /* Removed unused k */
	CADNeuralFace *face = &state->face;

	memset(face, 0, sizeof(CADNeuralFace));

	if (pthread_mutex_init(&face->mutex, NULL) != 0) {
		return 0;
	}

	face->neuron_count = NEURON_COUNT;
	face->vector_count = VECTOR_COUNT;

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

	printf("  ├─ Allocating vectors... ");
	fflush(stdout);

	face->vectors = (NeuralVectorLine*) aligned_alloc(64,
	VECTOR_COUNT * sizeof(NeuralVectorLine));
	if (!face->vectors) {
		free(face->neurons);
		pthread_mutex_destroy(&face->mutex);
		return 0;
	}
	memset(face->vectors, 0, VECTOR_COUNT * sizeof(NeuralVectorLine));
	printf("OK\n");

	printf("  ├─ Initializing neurons... ");
	fflush(stdout);

	/* Initialize neurons */
	for (i = 0; i < NEURON_COUNT; i++) {
		DimensionalNeuron *n = &face->neurons[i];

		/* Position on a sphere */
		float theta = (float) (i % 360) * 0.0174533f;
		float phi = (float) ((i / 360) % 180 - 90) * 0.0174533f;

		n->pos.x = (float) (cos(phi) * cos(theta) * 2.0);
		n->pos.y = (float) (cos(phi) * sin(theta) * 1.8);
		n->pos.z = (float) (sin(phi) * 2.4);
		n->pos.d = (float) (sin(theta) * 0.5);
		n->pos.a = (float) (cos(phi) * 0.5);
		n->pos.weight = 1.0f;

		n->potential = (float) rand() / RAND_MAX * 0.1f;
		n->threshold = 0.5f;
		n->refractory = 0.0f;
		n->adaptation = 0.1f;
		n->spikes = 0;
		n->id = i;

		for (j = 0; j < AXIS_COUNT; j++) {
			n->weights[j] = 0.5f + 0.5f * (float) rand() / RAND_MAX;
		}

		/* Color by position */
		n->color.r = (unsigned char) (128 + 127 * sin(n->pos.x));
		n->color.g = (unsigned char) (128 + 127 * cos(n->pos.y));
		n->color.b = (unsigned char) (128 + 127 * sin(n->pos.z));
		n->color.a = 200;
	}
	printf("OK\n");

	printf("  ├─ Generating vectors... ");
	fflush(stdout);

	/* Initialize vectors */
	for (i = 0; i < VECTOR_COUNT; i++) {
		unsigned int pre = (i * 2654435761u) % NEURON_COUNT;
		unsigned int post = (pre + 1 + (i % 100)) % NEURON_COUNT;
		unsigned int bundle = i % AXIS_COUNT;

		DimensionalNeuron *pre_n = &face->neurons[pre];
		DimensionalNeuron *post_n = &face->neurons[post];
		NeuralVectorLine *v = &face->vectors[i];

		v->start = pre_n->pos;
		v->end = post_n->pos;
		v->strength = 0.5f;
		v->bundle = bundle;

		/* Color by bundle */
		switch (bundle) {
		case 0:
			v->color.r = 255;
			v->color.g = 100;
			v->color.b = 100;
			break;
		case 1:
			v->color.r = 100;
			v->color.g = 255;
			v->color.b = 100;
			break;
		case 2:
			v->color.r = 100;
			v->color.g = 100;
			v->color.b = 255;
			break;
		case 3:
			v->color.r = 255;
			v->color.g = 255;
			v->color.b = 100;
			break;
		case 4:
			v->color.r = 255;
			v->color.g = 100;
			v->color.b = 255;
			break;
		}
		v->color.a = 100;
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

	face->center[0] = (min_x + max_x) * 0.5f;
	face->center[1] = (min_y + max_y) * 0.5f;
	face->center[2] = (min_z + max_z) * 0.5f;

	face->radius = (float) sqrt(
			(max_x - min_x) * (max_x - min_x)
					+ (max_y - min_y) * (max_y - min_y)
					+ (max_z - min_z) * (max_z - min_z)) * 0.5f;

	face->initialized = 1;

	printf("OK (radius: %.2f)\n", face->radius);

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
	if (face->vectors) {
		free(face->vectors);
		face->vectors = NULL;
	}
	pthread_mutex_unlock(&face->mutex);
	pthread_mutex_destroy(&face->mutex);
}

static void update_neurons(AppState *state) {
	unsigned int i;
	CADNeuralFace *face = &state->face;

	if (!face->initialized)
		return;

	pthread_mutex_lock(&face->mutex);

	for (i = 0; i < face->neuron_count; i++) {
		DimensionalNeuron *n = &face->neurons[i];

		/* Simple neuron update */
		float input = (n->pos.x * 0.1f + n->pos.y * 0.1f + n->pos.z * 0.1f);
		n->potential += input - n->adaptation * n->potential;

		if (n->potential > n->threshold && n->refractory < 0.1f) {
			n->spikes++;
			n->potential *= 0.1f;
			n->refractory = 1.0f;
		}

		if (n->refractory > 0.0f) {
			n->refractory -= 0.1f;
		}
	}

	pthread_mutex_unlock(&face->mutex);
}

static void* thread_worker(void *arg) {
	AppState *state = (AppState*) arg;

	while (state->running) {
		if (!state->paused) {
			update_neurons(state);
		}
		safe_sleep(10);
	}
	return NULL;
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
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE:
				state->running = 0;
				break;
			case SDLK_SPACE:
				state->paused = !state->paused;
				printf("%s\n", state->paused ? "PAUSED" : "RUNNING");
				break;
			case SDLK_UP:
				state->camera_dist -= 0.5f;
				if (state->camera_dist < 3.0f)
					state->camera_dist = 3.0f;
				break;
			case SDLK_DOWN:
				state->camera_dist += 0.5f;
				if (state->camera_dist > 15.0f)
					state->camera_dist = 15.0f;
				break;
			case SDLK_LEFT:
				state->camera_angle -= 0.1f;
				break;
			case SDLK_RIGHT:
				state->camera_angle += 0.1f;
				break;
			}
			break;

		case SDL_WINDOWEVENT:
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				g_gl.width = event.window.data1;
				g_gl.height = event.window.data2;
				glViewport(0, 0, g_gl.width, g_gl.height);
			}
			break;
		}
	}
}

static void signal_handler(int sig) {
	(void) sig;
	/* Signal handled in main loop */
}

/* ============================================================================
 * Main Function - Simplified for stability
 * =========================================================================== */

int main(int argc, char **argv) {
	AppState state;
	unsigned int i, current_time;

	(void) argc;
	(void) argv;

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	srand((unsigned int) time(NULL));

	/* Initialize state */
	memset(&state, 0, sizeof(AppState));
	state.running = 1;
	state.paused = 0;
	state.camera_angle = 0.0f;
	state.camera_dist = 8.0f;
	state.camera_height = 2.0f;
	state.last_frame.tv_sec = 0;
	state.last_frame.tv_nsec = 0;
	state.frame_count = 0;
	state.fps = 0.0;

	printf(
			"\n╔════════════════════════════════════════════════════════════╗\n");
	printf("║     MIMIX CAD Face - Neural Dimensional System v%s     ║\n",
			PROGRAM_VERSION);
	printf(
			"╚════════════════════════════════════════════════════════════╝\n\n");

	printf("System Configuration:\n");
	printf("  ├─ CPU Cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
	printf("  └─ Threads: %d\n\n", MAX_THREADS);

	printf("Generating Neural CAD face...\n");

	if (!init_neural_face(&state)) {
		fprintf(stderr, "Failed to initialize neural face\n");
		return 1;
	}

	printf("\n  ├─ Neurons: %d\n", state.face.neuron_count);
	printf("  ├─ Vectors: %d\n", state.face.vector_count);
	printf("  └─ Radius: %.2f\n\n", state.face.radius);

	printf("Initializing OpenGL... ");
	fflush(stdout);

	if (!init_opengl()) {
		fprintf(stderr, "OpenGL initialization failed\n");
		free_neural_face(&state);
		return 1;
	}
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

	printf("Controls:\n");
	printf("  ESC - Exit\n");
	printf("  SPACE - Pause\n");
	printf("  Arrow Keys - Camera\n\n");

	state.last_frame.tv_sec = get_time_ms() / 1000;
	state.last_frame.tv_nsec = 0;
	state.frame_count = 0;

	/* Main loop */
	while (state.running) {
		current_time = get_time_ms();

		handle_events(&state);

		/* Render frame */
		render_frame(&state);

		/* Update camera */
		if (!state.paused) {
			state.camera_angle += 0.01f;
		}

		/* FPS calculation */
		state.frame_count++;
		if (current_time - state.last_frame.tv_sec * 1000 >= 1000) {
			state.fps = (double) state.frame_count;
			printf("\rFPS: %.1f", state.fps);
			fflush(stdout);
			state.frame_count = 0;
			state.last_frame.tv_sec = current_time / 1000;
		}

		safe_sleep(FRAME_TIME_MS);
	}

	printf("\n\nShutting down...\n");

	/* Clean shutdown */
	state.running = 0;

	for (i = 0; i < state.thread_count; i++) {
		pthread_join(state.threads[i], NULL);
	}

	shutdown_opengl();
	free_neural_face(&state);
	SDL_Quit();

	printf("Shutdown complete. Goodbye.\n");

	return 0;
}
