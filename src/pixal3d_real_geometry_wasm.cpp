// Real-input browser E2E up to the current WebGPU texture-conditioning boundary:
// RGBA images + transforms.json -> live SS -> live Shape512 -> live Shape1024 -> shape decode -> GLB.
// This is deliberately named geometry E2E, not full E2E: NAF@1024 texture conditioning remains
// the single known blocker for image->textured-GLB on WebGPU.
#include "pixal3d_input.h"
#include "pixal3d_cond.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PXEXP EMSCRIPTEN_KEEPALIVE
#else
#define PXEXP
#endif
using std::array;using std::string;using std::vector;using namespace trellis;
namespace {
string report;void rlog(const char*f,...){char b[1024];va_list a;va_start(a,f);vsnprintf(b,sizeof b,f,a);va_end(a);report+=b;fputs(b,stdout);fflush(stdout);} 
static const float MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
vector<float> noise(size_t n,uint32_t seed){std::mt19937 g(seed);std::normal_distribution<float>d(0,1);vector<float>x(n);for(auto&v:x)v=d(g);return x;}
vector<float> sflow(const string&p,const vector<array<int,3>>&co,Pixal3dCond c,vector<float>nz){Model m=Model::load(p,0);DiTParams q;q.in_ch=32;q.out_ch=32;q.d_cond=1024;if(!dit_detect_proj_attn(m,q)||q.d_proj!=2048)throw std::runtime_error("bad shape flow");auto*r=make_sparse_runner(m,q,co,c.n_global);vector<float>ng(c.global.size(),0),np(c.proj.size(),0);FlowFwdProj f=[&](const vector<float>&x,float t,const float*a,const float*b){return r->forward(x,t,a,b);};SamplerParams sp;sp.steps=12;sp.guidance_strength=7.5f;sp.guidance_rescale=.5f;sp.gi0=.6f;sp.gi1=1;sp.rescale_t=3;auto o=sample_flow(f,nz,c.global.data(),ng.data(),c.proj.data(),np.data(),sp);delete r;m.free();return o;}
int run(const vector<string>&m,const string&views,const string&out,uint32_t seed){Pixal3dInputViews in;string err;if(!pixal3d_load_input_views(views,in,err)){rlog("input error: %s\n",err.c_str());return 2;}rlog("real input: V=%zu mesh_scale=%.4f\n",in.views512.size(),in.mesh_scale);
 Pixal3dCond cs;{Model d=Model::load(m[0],0);cs=pixal3d_cond_ss_gpu(d,in.views512,512,16,in.mesh_scale);d.free();}vector<float>ng(cs.global.size(),0),np(cs.proj.size(),0);Model sm=Model::load(m[2],0);DiTParams dp;dp.in_ch=8;dp.out_ch=8;dp.d_cond=1024;if(!dit_detect_proj_attn(sm,dp))return 3;auto*dr=make_dense_runner(sm,dp,16,cs.n_global);FlowFwdProj ff=[&](const vector<float>&x,float t,const float*a,const float*b){return dr->forward(x,t,a,b);};SamplerParams s;s.steps=12;s.guidance_strength=7.5f;s.guidance_rescale=.7f;s.gi0=.6f;s.gi1=1;s.rescale_t=5;auto z=sample_flow(ff,noise(8*4096,seed),cs.global.data(),ng.data(),cs.proj.data(),np.data(),s);delete dr;sm.free();vector<float>zd(8*4096);for(int c=0;c<8;++c)for(int i=0;i<4096;++i)zd[(size_t)c*4096+i]=z[c+8*i];vector<array<int,3>>co;{Model d=Model::load(m[3],0);auto l=ss_decode(d,zd);d.free();co=ss_coords(l,64,32);}if(co.empty())return 4;
 Pixal3dCond c5;{Model d=Model::load(m[0],0),n=Model::load(m[1],0);c5=pixal3d_cond_slat_gpu(d,n,in.views512,{512,32,512,in.mesh_scale});d.free();n.free();}c5.proj=pixal3d_gather_proj(c5.proj,32,c5.d_proj,co);auto ln=sflow(m[4],co,c5,noise(32*co.size(),seed+1));vector<float>ld(ln.size());for(size_t i=0;i<co.size();++i)for(int c=0;c<32;++c)ld[c+32*i]=ln[c+32*i]*STD[c]+MEAN[c];vector<array<int,3>>up;{Model d=Model::load(m[5],0);up=shape_upsample(d,ld,co);d.free();}std::set<array<int,3>>q;for(auto&c:up)q.insert({(int)std::lround((c[0]+.5f)/512.f*63),(int)std::lround((c[1]+.5f)/512.f*63),(int)std::lround((c[2]+.5f)/512.f*63)});vector<array<int,3>>hr(q.begin(),q.end());rlog("live Shape1024 tokens=%zu\n",hr.size());
 Pixal3dCond ch;{Model d=Model::load(m[0],0),n=Model::load(m[1],0);ch=pixal3d_cond_slat_gpu(d,n,in.views1024,{1024,64,512,in.mesh_scale});d.free();n.free();}ch.proj=pixal3d_gather_proj(ch.proj,64,ch.d_proj,hr);auto hn=sflow(m[6],hr,ch,noise(32*hr.size(),seed+2));vector<float>hd(hn.size());for(size_t i=0;i<hr.size();++i)for(int c=0;c<32;++c)hd[c+32*i]=hn[c+32*i]*STD[c]+MEAN[c];ShapeOut so;{Model d=Model::load(m[5],0);so=shape_decode(d,hd,hr,1024);d.free();}Mesh mesh=dual_grid_to_mesh(so);if(mesh.F()<=0)return 5;if(!write_glb(out.c_str(),mesh.verts.data(),mesh.V(),mesh.faces.data(),mesh.F()))return 6;rlog("REAL_INPUT_LIVE_SS=1\nREAL_INPUT_LIVE_SHAPE512=1\nREAL_INPUT_LIVE_SHAPE1024=1\nREAL_INPUT_TEXTURE_COND_BLOCKED=1\nREAL_GEOMETRY_E2E_RESULT: OK V=%d F=%d\n",mesh.V(),mesh.F());return 0;}
}
extern "C" PXEXP const char* pixal3d_real_geometry_run(const char*dino,const char*naf,const char*ssf,const char*ssd,const char*s512,const char*sdec,const char*s1024,const char*views,const char*out,int seed){report.clear();try{int rc=run({dino,naf,ssf,ssd,s512,sdec,s1024},views?views:"",out?out:"/out/real_geometry.glb",seed?seed:1);rlog(rc==0?"REAL_GEOMETRY_RESULT: OK\n":"REAL_GEOMETRY_RESULT: FAIL rc=%d\n",rc);}catch(const std::exception&e){rlog("REAL_GEOMETRY_RESULT: EXCEPTION %s\n",e.what());}return report.c_str();}
