#include "lab2.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
#define IX(i,j) ((i)+(N+2)*(j))
#include "solver.c"
/* external definitions (from solver.c) */
//extern void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
//extern void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );
/* global variables */
static int N = 64;
static float dt=0.1f, diff=0.0f, visc=0.0f;
static float force=5.0f, source=1500.0f;
static int dvel;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;
int convert(int r,int c) {
	return c * W + r;
}
void init() {
	int size = (N+2)*(N+2);
	u			= (float *) malloc ( size*sizeof(float) );
	v			= (float *) malloc ( size*sizeof(float) );
	u_prev		= (float *) malloc ( size*sizeof(float) );
	v_prev		= (float *) malloc ( size*sizeof(float) );
	dens		= (float *) malloc ( size*sizeof(float) );
	dens_prev	= (float *) malloc ( size*sizeof(float) );

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
	}
	//add density:

  int i, j;

  for ( i=0 ; i<size ; i++ ) {
      u[i] = v[i] = dens_prev[i] = 0.0f;
  }

}
void animate_parameter(float * d, float * u, float * v,int xx,int yy, int forcex, int forcey) {
	int i, j, size = (N+2)*(N+2);

    for ( i=0 ; i<size ; i++ ) {
        u[i] = v[i] = d[i] = 0.0f;
    }
		if(xx==0||yy==0) return;

    i = (int)((       xx /(float)W)*N+1);
    j = (int)(((H-yy)/(float)H)*N+1);

    if ( i<1 || i>N || j<1 || j>N ) return;
    u[IX(i,j)] = force * forcex;
    v[IX(i,j)] = force * forcey;

    d[IX(i,j)] = source;
    return;
}
void animate(int t) {
	if (t >= 1 && t<=2){
		animate_parameter ( dens_prev, u_prev, v_prev, 100,100,50,50 );
	} else if (t >= 20 && t < 80) {
		animate_parameter ( dens_prev, u_prev, v_prev, 220,250,30,-20-(t-30) );
	} else if (t >= 120 && t < 150) {
		animate_parameter ( dens_prev, u_prev, v_prev, 220,230,30,-20-(t-30) );
	}else if (t >= 180 && t<= 190){
		animate_parameter ( dens_prev, u_prev, v_prev, 220,230,30,-20-(t-30) );
	}
	else {
		animate_parameter ( dens_prev, u_prev, v_prev, 0,0,10,10 );
	}
  vel_step ( N, u, v, u_prev, v_prev, visc, dt );
  dens_step ( N, dens, dens_prev, u, v, diff, dt );
}
void add_force(int xx, int yy){
	int i,j;
	i = (int)((       xx /(float)W)*N+1);
	j = (int)(((H-yy)/(float)H)*N+1);

	if ( i<1 || i>N || j<1 || j>N ) return;
	u[IX(i,j)] = force * 10;
	v[IX(i,j)] = force * 10;
}
void add_dens(int mx, int my) {
	int i = (int)((       mx /(float)W)*N+1);
  int j = (int)(((H-my)/(float)H)*N+1);
	dens_prev[IX(i,j)] = source;
	printf("add: %d,%d,%f\n",i,j,dens_prev[IX(i,j)]);

}
struct Lab2VideoGenerator::Impl {
	int t = 0;
	int posx = 0;
	int posy=0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};
void drawPoint(uint8_t *yuv, int x, int y,float color) {
	float RGBcolor = color*25500 > 255 ? 255: color*25500;
	//printf("%f ",RGBcolor);
	for(int i = y; i < y + 2; i++) {
		int pos = convert(x, i);
		cudaMemset(yuv + pos, int(RGBcolor), 2);
	}
}
void drawDensity(uint8_t *yuv) {
	int i, j;
	float x, y, h, d00, d01, d10, d11;
	h = 1.0f/N;
	for ( i=0 ; i<=N ; i++ ) {
		x = (i-0.5f)*h;
		for ( j=0 ; j<=N ; j++ ) {
			y = (j-0.5f)*h;

			d00 = dens[IX(i,j)];
			d01 = dens[IX(i,j+1)];
			d10 = dens[IX(i+1,j)];
			d11 = dens[IX(i+1,j+1)];
			drawPoint(yuv, int(x*W), H-int(y*H), d00);
			drawPoint(yuv, int((x+h)*W), H-int(y*H), d10);
			drawPoint(yuv, int((x+h)*W), H-int((y+h)*H), d11);
			drawPoint(yuv, int(x*W), H-int((y+h)*H), d01);


		}
	}
}
void changeDens(){
	int i,j;
	int index = 30;
	for(i=10;i<42;i++){
		dens[IX(index,i)] = 0.0f;
	}
	index = 20;
	for(i=index;i<30;i++){
		dens[IX(i,10)] = 0.0f;
	}
	for(i=index;i<30;i++){
		dens[IX(i,25)] = 0.0f;
	}
	for(i=index;i<30;i++){
		dens[IX(i,40)] = 0.0f;
	}

}
void changeDens2(){
	int i,j;
	int index = 30;
	for(i=24;i<42;i++){
		dens[IX(index,i)] = 0.0f;
	}
	index = 20;
	for(i=10;i<24;i++){
		dens[IX(index,i)] = 0.0f;
	}
	for(i=index;i<30;i++){
		dens[IX(i,10)] = 0.0f;
	}
	for(i=index;i<30;i++){
		dens[IX(i,25)] = 0.0f;
	}
	for(i=index;i<30;i++){
		dens[IX(i,40)] = 0.0f;
	}

}
void changeDens1(){
	int i,j;
	int index = 30;
	for(i=10;i<42;i++){
		dens[IX(index,i)] = 0.0f;
	}
}
void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	if (impl->t == 0) {
		cudaMemset(yuv,0,W*H);
		init();
	}
		/*for(int i=0;i<(N+2)*(N+2);i++){
			printf("%f ",dens[i]);
		}
		printf("===========================\n");*/
	printf("%d\n",impl->t);

	if(impl->t < 120){
		changeDens();
	} else if (impl->t < 180) {
		changeDens2();
	} else {
		changeDens1();
	}
	drawDensity(yuv);
	animate(impl->t);
	//}
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}
