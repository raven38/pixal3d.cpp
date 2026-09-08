// Real-input browser E2E.
//
//  * pixal3d_real_geometry_run: RGBA images + transforms.json -> live SS -> live Shape512
//    -> live Shape1024 -> shape decode -> geometry GLB（軽い gate として残す）。
//  * pixal3d_real_full_run: 上に live Texture-1024 conditioning -> Texture Flow ->
//    Shape/Texture decode -> production postprocess -> textured GLB を足した true full E2E。
//    Texture conditioning は分割グラフ版 pixal3d_cond_slat_gpu（sparse coords）で、
//    NAF@1024 の [1024, 1024^2] を一度に materialize しない（docs/PIXAL3D_WEBGPU_MEMORY.md）。
#include "pixal3d_input.h"
#include "pixal3d_cond.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "pixal3d_postprocess.h"
#ifndef __EMSCRIPTEN__
#include "npy.h"
#endif
#include "ggml-backend.h"
#include "trellis_args.h"
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <malloc.h>
#define PXEXP EMSCRIPTEN_KEEPALIVE
#else
#define PXEXP
#endif
using std::array;using std::string;using std::vector;using namespace trellis;
namespace {
string g_dump_fixture;   // native の --dump-fixture 用（browser では常に空）
string report;void rlog(const char*f,...){char b[1024];va_list a;va_start(a,f);vsnprintf(b,sizeof b,f,a);va_end(a);report+=b;fputs(b,stdout);fflush(stdout);} 
// wasm32 の 4 GiB ヒープの内訳を段ごとに残す（内訳の読み方は full_e2e 側の fheap を見よ）。
void rheap(const char*tag){
#ifdef __EMSCRIPTEN__
 struct mallinfo mi=mallinfo();
 rlog("[heap] %-18s grown %.0f MB  live %.0f MB  free %.0f MB\n",tag,
      emscripten_get_heap_size()/1048576.0,
      (double)(unsigned)mi.uordblks/1048576.0,(double)(unsigned)mi.fordblks/1048576.0);
#else
 (void)tag;
#endif
}
static const float MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
static const float TMEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TSTD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};
vector<float> noise(size_t n,uint32_t seed){std::mt19937 g(seed);std::normal_distribution<float>d(0,1);vector<float>x(n);for(auto&v:x)v=d(g);return x;}
vector<float> sflow(const string&p,const vector<array<int,3>>&co,Pixal3dCond c,vector<float>nz){Model m=Model::load(p,0);DiTParams q;q.in_ch=32;q.out_ch=32;q.d_cond=1024;if(!dit_detect_proj_attn(m,q)||q.d_proj!=2048)throw std::runtime_error("bad shape flow");auto*r=make_sparse_runner(m,q,co,c.n_global);vector<float>ng(c.global.size(),0),np(c.proj.size(),0);FlowFwdProj f=[&](const vector<float>&x,float t,const float*a,const float*b){return r->forward(x,t,a,b);};SamplerParams sp;sp.steps=12;sp.guidance_strength=7.5f;sp.guidance_rescale=.5f;sp.gi0=.6f;sp.gi1=1;sp.rescale_t=3;auto o=sample_flow(f,nz,c.global.data(),ng.data(),c.proj.data(),np.data(),sp);delete r;m.free();return o;}
// Texture Flow: 入力は [shape_norm || x] の 64ch（full_e2e と同じ順・同じサンプラ設定）。
vector<float> tflow(const string&p,const vector<array<int,3>>&co,const Pixal3dCond&c,const vector<float>&shape_norm,const vector<float>&nz){
 const int N=(int)co.size();Model m=Model::load(p,0);DiTParams q;q.in_ch=64;q.out_ch=32;q.d_cond=1024;
 if(!dit_detect_proj_attn(m,q)||q.d_proj!=2048)throw std::runtime_error("bad texture flow checkpoint");
 auto*r=make_sparse_runner(m,q,co,c.n_global);vector<float>ng(c.global.size(),0),np(c.proj.size(),0),x64((size_t)64*N);
 FlowFwdProj f=[&](const vector<float>&st,float t,const float*a,const float*b){for(int n=0;n<N;++n){for(int k=0;k<32;++k)x64[k+64*n]=st[k+32*n];for(int k=0;k<32;++k)x64[32+k+64*n]=shape_norm[k+32*n];}return r->forward(x64,t,a,b);};
 SamplerParams sp;sp.steps=12;sp.guidance_strength=1.f;sp.guidance_rescale=0;sp.gi0=.6f;sp.gi1=.9f;sp.rescale_t=3;
 auto o=sample_flow(f,nz,c.global.data(),ng.data(),c.proj.data(),np.data(),sp);delete r;m.free();return o;}


// SS decoder だけは backend を分ける。ggml WebGPU backend は GGML_OP_IM2COL_3D も
// GGML_OP_CONV_3D も実装していない（docs/PIXAL3D_WEBGPU_OP_GAP.md）。エラーにならず
// NaN 混じりの logits が返るため、SS の active voxel が潰れて以降の形状が全部壊れる
// （実測: browser の ss_logits が nonfinite=21143、ss_coords の bbox が x[2..4] に収縮）。
// trellis-test-pixal3d-ss-sample の dec_gpu=-1 と同じ扱いで CPU backend に載せる。
constexpr int SS_DEC_BACKEND =
#ifdef __EMSCRIPTEN__
    -1;   // CPU
#else
    0;
#endif

// ---- 段ごとの計測（browser と native の最初の食い違いを特定するため）----
// 形が壊れたときに「どの段までは同じか」を数値で言えるようにする。座標は bbox と重心、
// テンソルは mean/std/min/max と非有限数。両バックエンドで同じ行が出る。
void log_coords(const char* tag,const vector<array<int,3>>&c){
 if(c.empty()){rlog("[stage] %-14s N=0\n",tag);return;}
 int lo[3]={1<<30,1<<30,1<<30},hi[3]={-(1<<30),-(1<<30),-(1<<30)};double mu[3]={0,0,0};
 for(const auto&v:c)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],v[k]);hi[k]=std::max(hi[k],v[k]);mu[k]+=v[k];}
 rlog("[stage] %-14s N=%zu bbox x[%d..%d] y[%d..%d] z[%d..%d] centroid(%.1f,%.1f,%.1f)\n",
      tag,c.size(),lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],mu[0]/c.size(),mu[1]/c.size(),mu[2]/c.size());}
void log_vec(const char* tag,const vector<float>&v){
 if(v.empty()){rlog("[stage] %-14s empty\n",tag);return;}
 double s1=0,s2=0;float lo=v[0],hi=v[0];size_t nf=0;
 for(float x:v){if(!std::isfinite(x)){++nf;continue;}s1+=x;s2+=(double)x*x;lo=std::min(lo,x);hi=std::max(hi,x);}
 const double n=(double)(v.size()-nf),mean=n>0?s1/n:0.0;
 rlog("[stage] %-14s n=%zu mean=%+.6f std=%.6f min=%+.4f max=%+.4f nonfinite=%zu\n",
      tag,v.size(),mean,n>0?std::sqrt(std::max(0.0,s2/n-mean*mean)):0.0,lo,hi,nf);}
// 頂点配列の bbox（最終メッシュがどの軸で潰れているかを段の直後に見る）
void log_verts(const char* tag,const vector<float>&v){
 if(v.size()<3){rlog("[stage] %-14s no verts\n",tag);return;}
 float lo[3]={v[0],v[1],v[2]},hi[3]={v[0],v[1],v[2]};
 for(size_t i=0;i+2<v.size();i+=3)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],v[i+k]);hi[k]=std::max(hi[k],v[i+k]);}
 rlog("[stage] %-14s V=%zu bbox size (%.4f, %.4f, %.4f)\n",tag,v.size()/3,hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);}
// m: dino, naf, ss_flow, ss_dec, shape512, shape_dec, shape1024 [, tex_flow, tex_dec]
// tex 付き（m.size()==9）なら live Texture conditioning 以降まで回して textured GLB を書く。
int run(const vector<string>&m,const string&views,const string&out,uint32_t seed){
 const bool with_tex=(m.size()>=9);
#ifdef __EMSCRIPTEN__
 g_no_fa=true;   // WebGPU backend に BF16 K/V FlashAttention は無い（--no-fa 相当の exact SDPA）
#else
 if(getenv("TRELLIS_NOFA")) g_no_fa=true;   // native から browser と同じ経路を踏むとき
#endif
 Pixal3dInputViews in;string err;if(!pixal3d_load_input_views(views,in,err)){rlog("input error: %s\n",err.c_str());return 2;}rlog("real input: V=%zu mesh_scale=%.4f\n",in.views512.size(),in.mesh_scale);
 Pixal3dCond cs;{Model d=Model::load(m[0],0);cs=pixal3d_cond_ss_gpu(d,in.views512,512,16,in.mesh_scale);d.free();}log_vec("ss_cond_glob",cs.global);log_vec("ss_cond_proj",cs.proj);vector<float>ng(cs.global.size(),0),np(cs.proj.size(),0);Model sm=Model::load(m[2],0);DiTParams dp;dp.in_ch=8;dp.out_ch=8;dp.d_cond=1024;if(!dit_detect_proj_attn(sm,dp))return 3;auto*dr=make_dense_runner(sm,dp,16,cs.n_global);FlowFwdProj ff=[&](const vector<float>&x,float t,const float*a,const float*b){return dr->forward(x,t,a,b);};SamplerParams s;s.steps=12;s.guidance_strength=7.5f;s.guidance_rescale=.7f;s.gi0=.6f;s.gi1=1;s.rescale_t=5;auto z=sample_flow(ff,noise(8*4096,seed),cs.global.data(),ng.data(),cs.proj.data(),np.data(),s);delete dr;sm.free();log_vec("ss_latent",z);vector<float>zd(8*4096);for(int c=0;c<8;++c)for(int i=0;i<4096;++i)zd[(size_t)c*4096+i]=z[c+8*i];vector<array<int,3>>co;{Model d=Model::load(m[3],SS_DEC_BACKEND);auto l=ss_decode(d,zd);d.free();log_vec("ss_logits",l);co=ss_coords(l,64,32);}if(co.empty())return 4;log_coords("ss_coords@64",co);
 Pixal3dCond c5;{Model d=Model::load(m[0],0),n=Model::load(m[1],0);c5=pixal3d_cond_slat_gpu(d,n,in.views512,{512,32,512,in.mesh_scale},nullptr,&co);d.free();n.free();}log_vec("s512_cond_glob",c5.global);log_vec("s512_cond_proj",c5.proj);auto ln=sflow(m[4],co,c5,noise(32*co.size(),seed+1));log_vec("s512_latent",ln);vector<float>ld(ln.size());for(size_t i=0;i<co.size();++i)for(int c=0;c<32;++c)ld[c+32*i]=ln[c+32*i]*STD[c]+MEAN[c];vector<array<int,3>>up;{Model d=Model::load(m[5],0);up=shape_upsample(d,ld,co);d.free();}log_coords("upsample@512",up);std::set<array<int,3>>q;for(auto&c:up)q.insert({(int)std::lround((c[0]+.5f)/512.f*63),(int)std::lround((c[1]+.5f)/512.f*63),(int)std::lround((c[2]+.5f)/512.f*63)});vector<array<int,3>>hr(q.begin(),q.end());rlog("live Shape1024 tokens=%zu\n",hr.size());log_coords("hr_coords@64",hr);
 // hr が確定したら SS / Shape512 側は一切読まない。q は node-based container で vector より重い。
 std::set<array<int,3>>().swap(q);up.clear();up.shrink_to_fit();
 cs=Pixal3dCond{};c5=Pixal3dCond{};ng.clear();ng.shrink_to_fit();np.clear();np.shrink_to_fit();
 z.clear();z.shrink_to_fit();zd.clear();zd.shrink_to_fit();ln.clear();ln.shrink_to_fit();ld.clear();ld.shrink_to_fit();
 in.views512.clear();in.views512.shrink_to_fit();rheap("after Shape512");
 Pixal3dCond ch;{Model d=Model::load(m[0],0),n=Model::load(m[1],0);ch=pixal3d_cond_slat_gpu(d,n,in.views1024,{1024,64,512,in.mesh_scale},nullptr,&hr);d.free();n.free();}log_vec("s1024_cond_glob",ch.global);log_vec("s1024_cond_proj",ch.proj);auto hn=sflow(m[6],hr,ch,noise(32*hr.size(),seed+2));log_vec("s1024_latent",hn);vector<float>hd(hn.size());for(size_t i=0;i<hr.size();++i)for(int c=0;c<32;++c)hd[c+32*i]=hn[c+32*i]*STD[c]+MEAN[c];ch=Pixal3dCond{};rheap("after Shape1024");
 ShapeOut so;{Model d=Model::load(m[5],0);so=shape_decode(d,hd,hr,1024);d.free();}rlog("shape_decode done: %zu voxels res=%d\n",so.coords.size(),so.res);rheap("shape_decode");log_coords("decoded_vox",so.coords);Mesh mesh=dual_grid_to_mesh(so);if(mesh.F()<=0)return 5;
 // feats7 は [7,M]（M=4.76M で約 133 MB）。メッシュを作ったら tex decoder には要らない。
 so.feats7.clear();so.feats7.shrink_to_fit();rlog("mesh V=%d F=%d\n",mesh.V(),mesh.F());rheap("dual_grid_to_mesh");log_verts("raw_mesh",mesh.verts);
 if(!with_tex){if(!write_glb(out.c_str(),mesh.verts.data(),mesh.V(),mesh.faces.data(),mesh.F()))return 6;rlog("REAL_INPUT_LIVE_SS=1\nREAL_INPUT_LIVE_SHAPE512=1\nREAL_INPUT_LIVE_SHAPE1024=1\nREAL_INPUT_TEXTURE_COND_BLOCKED=1\nREAL_GEOMETRY_E2E_RESULT: OK V=%d F=%d\n",mesh.V(),mesh.F());return 0;}
 // ---- live Texture-1024 conditioning（fixture 注入なし）----
 Pixal3dCond ct;{Model d=Model::load(m[0],0),n=Model::load(m[1],0);Pixal3dCondStats cst;ct=pixal3d_cond_slat_gpu(d,n,in.views1024,{1024,64,1024,in.mesh_scale},&cst,&hr);d.free();n.free();
  rlog("live tex cond: tokens=%zu d_proj=%d  graph peak %.1f MB, resident %.1f MB, %.1f s\n",hr.size(),ct.d_proj,cst.view_alloc_bytes/1048576.0,(cst.weight_bytes+cst.cond_bytes)/1048576.0,cst.total_ms/1000.0);}
#ifndef __EMSCRIPTEN__
 // Shape-1024 fixture を書き出しておくと、browser の partial E2E
 // (Shape1024 fixture -> Texture Flow -> decode -> GLB) を実データで回せる。
 if(!g_dump_fixture.empty()){const int64_t N=(int64_t)hr.size();
  vector<int32_t> co3((size_t)N*3);for(int64_t i=0;i<N;++i){co3[i*3]=hr[i][0];co3[i*3+1]=hr[i][1];co3[i*3+2]=hr[i][2];}
  npy::save_i32(g_dump_fixture+"/hr_coords.npy",co3.data(),{N,3});
  npy::save(g_dump_fixture+"/f32_shape_slat.npy",hd.data(),{N,32});
  npy::save(g_dump_fixture+"/f32_tex_concat_cond.npy",hn.data(),{N,32});
  npy::save(g_dump_fixture+"/tex_cond_global.npy",ct.global.data(),{1,5,1024});
  npy::save(g_dump_fixture+"/tex_cond_proj.npy",ct.proj.data(),{N,(int64_t)ct.d_proj});
  auto tnz=noise(32*(size_t)N,seed+3);npy::save(g_dump_fixture+"/tex_noise.npy",tnz.data(),{N,32});
  npy::save(g_dump_fixture+"/tex_norm_mean.npy",TMEAN,{32});npy::save(g_dump_fixture+"/tex_norm_std.npy",TSTD,{32});
  rlog("dumped Shape-1024 fixture (N=%lld) to %s\n",(long long)N,g_dump_fixture.c_str());}
#endif
 // Texture Flow の入力は hn（shape_norm）。ここまでに使い終わったものを落としてから入る。
 // hd / hn は上の native fixture dump で読むので、解放はその後ろに置く。
 hd.clear();hd.shrink_to_fit();rheap("before TextureFlow");
 auto tn=tflow(m[7],hr,ct,hn,noise(32*hr.size(),seed+3));
 // ct.proj は N*2048*4。Texture Flow を抜けたら conditioning も 1024 の view も不要。
 ct=Pixal3dCond{};hn.clear();hn.shrink_to_fit();in.views1024.clear();in.views1024.shrink_to_fit();rheap("after TextureFlow");vector<float>td(tn.size());for(size_t i=0;i<hr.size();++i)for(int c=0;c<32;++c)td[c+32*i]=tn[c+32*i]*TSTD[c]+TMEAN[c];
 vector<float>raw;{Model d=Model::load(m[8],0);raw=tex_decode(d,td,hr,so.subs);d.free();}if(raw.size()!=so.coords.size()*6)return 7;rlog("tex_decode done: %zu voxels x6 (%.1f MB)\n",so.coords.size(),raw.size()*4/1048576.0);rheap("tex_decode");
 rlog("tex decode: %zu voxels x6 (%.1f MB), raw mesh V=%d F=%d\n",so.coords.size(),raw.size()*4/1048576.0,mesh.V(),mesh.F());
 vector<float>pbr(raw.size());for(size_t i=0;i<raw.size();++i)pbr[i]=std::clamp(.5f*raw[i]+.5f,0.f,1.f);
 Pixal3dPostprocessOptions opt;opt.remesh_band=1;opt.use_xatlas=false;opt.use_webp=false;opt.texture_size=4096;
#ifdef __EMSCRIPTEN__
 // wasm32 のヒープは 4 GiB が上限。res=1024 の narrow-band remesh は実測 7.8M 頂点 /
 // 15.6M 面を作り、その後の QEM と合わせて収まらない（std::bad_alloc）。粗いグリッドで
 // 同じ経路を回す。使った値は postprocess の report 行に出る。
 opt.remesh_res=512;opt.target_faces=500000;
#else
 opt.target_faces=1000000;
#endif
 string pr;
 if(!pixal3d_write_production_glb(out,mesh,so.coords,pbr,so.res,opt,&pr)){rlog("%s",pr.c_str());return 8;}rlog("%s",pr.c_str());
 FILE*f=fopen(out.c_str(),"rb");long bytes=-1;if(f){fseek(f,0,SEEK_END);bytes=ftell(f);fclose(f);}
 rlog("REAL_INPUT_LIVE_SS=1\nREAL_INPUT_LIVE_SHAPE512=1\nREAL_INPUT_LIVE_SHAPE1024=1\nREAL_INPUT_LIVE_TEXTURE_COND=1\nREAL_INPUT_TEXTURE_COND_BLOCKED=0\nREAL_FULL_E2E_RESULT: OK V=%d F=%d glb_bytes=%ld atlas=%d\n",mesh.V(),mesh.F(),bytes,opt.texture_size);return 0;}
}
extern "C" PXEXP const char* pixal3d_real_full_run(const char*dino,const char*naf,const char*ssf,const char*ssd,const char*s512,const char*sdec,const char*s1024,const char*texf,const char*texd,const char*views,const char*out,int seed){report.clear();try{int rc=run({dino,naf,ssf,ssd,s512,sdec,s1024,texf,texd},views?views:"",out?out:"/out/real_full.glb",seed?seed:1);rlog(rc==0?"REAL_FULL_RESULT: OK\n":"REAL_FULL_RESULT: FAIL rc=%d\n",rc);}catch(const std::exception&e){rlog("REAL_FULL_RESULT: EXCEPTION %s\n",e.what());}return report.c_str();}

extern "C" PXEXP const char* pixal3d_real_geometry_run(const char*dino,const char*naf,const char*ssf,const char*ssd,const char*s512,const char*sdec,const char*s1024,const char*views,const char*out,int seed){report.clear();try{int rc=run({dino,naf,ssf,ssd,s512,sdec,s1024},views?views:"",out?out:"/out/real_geometry.glb",seed?seed:1);rlog(rc==0?"REAL_GEOMETRY_RESULT: OK\n":"REAL_GEOMETRY_RESULT: FAIL rc=%d\n",rc);}catch(const std::exception&e){rlog("REAL_GEOMETRY_RESULT: EXCEPTION %s\n",e.what());}return report.c_str();}

#ifndef __EMSCRIPTEN__
// native ドライバ（browser へ投げる前に同一 C++ 経路を Metal/CUDA で通すための dry-run）。
//   trellis-test-pixal3d-real-e2e <dino> <naf> <ss_flow> <ss_dec> <shape512> <shape_dec>
//                                 <shape1024> <views_dir> <out.glb> [seed] [tex_flow] [tex_dec]
int main(int argc,char**argv){
    if(argc<10){fprintf(stderr,"usage: %s <dinov3.gguf> <pixal3d_naf.gguf> <pixal3d_ss_flow_mv.gguf> <ss_dec.gguf>"
        " <pixal3d_shape_flow_512_mv.gguf> <shape_dec.gguf> <pixal3d_shape_flow_1024_mv.gguf>"
        " <views_dir> <out.glb> [seed] [pixal3d_tex_flow_1024_mv.gguf] [tex_dec.gguf]\n",argv[0]);return 2;}
    const int seed=argc>10?atoi(argv[10]):1;
    if(const char* d=getenv("PIXAL3D_DUMP_FIXTURE")) g_dump_fixture=d;   // Shape-1024 fixture の書き出し先
    const char* r = (argc>12)
        ? pixal3d_real_full_run(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[11],argv[12],argv[8],argv[9],seed)
        : pixal3d_real_geometry_run(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8],argv[9],seed);
    const bool ok = strstr(r,"RESULT: OK") != nullptr;
    printf("\n%s\n", ok?"REAL_E2E: PASS":"REAL_E2E: FAIL");
    return ok?0:1;
}
#endif
