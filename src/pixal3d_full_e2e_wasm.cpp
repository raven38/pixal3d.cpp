// Pixal3D browser full-model E2E orchestration.
//
// Phase 1 implemented here:
//   fixture views -> SS cond/flow/decode -> Shape512 cond/flow -> upsample/quantize
//   -> Shape1024 cond/flow -> Texture flow -> shape/texture decode -> GLB.
//
// Texture-1024 conditioning は分割グラフ版 pixal3d_cond_slat_gpu（sparse coords）で live に
// 計算する。NAF@1024 の [1024, 1024^2]（4 GiB）を一度に materialize せず、block chunk ごとに
// projection tap を回収するので WebGPU の maxBufferSize を踏まない
// （docs/PIXAL3D_WEBGPU_MEMORY.md）。fixture 注入は残っていない。
// PIXAL3D_TEX_COND_FIXTURE=1 を渡したときだけ、旧 fixture 経路を debug 用に使う。

#include "pixal3d_cascade.h"
#include "pixal3d_cond.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "dit.h"
#include "ss_decoder.h"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "mesh_glb.h"
#include "pixal3d_postprocess.h"
#include "npy.h"
#include "ggml-backend.h"
#include "trellis_args.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/heap.h>
#include <malloc.h>
#define PIXAL3D_FULL_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define PIXAL3D_FULL_EXPORT
#endif

using std::array; using std::string; using std::vector;
using namespace trellis;

namespace {
// fixture 駆動の full E2E は reference fixture（ss_res=32）の noise/coords に合わせて固定。
constexpr int kSsRes = 32;
string g_full_report;
void frep(const char* fmt, ...) {
    char b[2048]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    g_full_report += b; fputs(b, stdout); fflush(stdout);
}
// wasm32 の 4 GiB ヒープの内訳を段ごとに残す。native では何も出さない。
// grown は emscripten_get_heap_size()、つまり **一度 grow した linear memory のサイズ**で、
// free しても縮まない high-water mark。live/free は mallinfo() の実測値で、こちらが
// 「いま本当に確保されている量」。bad_alloc は live + 断片化 + 単発の巨大要求が
// 4 GiB に当たって起きるので、grown だけを見て残量を語らないこと。
// mallinfo の各フィールドは int なので 2 GB を超えると負になる。unsigned で読み直す。
void fheap(const char* tag) {
#ifdef __EMSCRIPTEN__
    struct mallinfo mi = mallinfo();
    frep("[heap] %-18s grown %.0f MB  live %.0f MB  free %.0f MB\n", tag,
         emscripten_get_heap_size() / 1048576.0,
         (double)(unsigned)mi.uordblks / 1048576.0, (double)(unsigned)mi.fordblks / 1048576.0);
#else
    (void)tag;
#endif
}
bool fexists(const string& p) { FILE* f=fopen(p.c_str(),"rb"); if(!f) return false; fclose(f); return true; }

static const float SHAPE_MEAN[32]={0.781296f,0.018091f,-0.495192f,-0.558457f,1.060530f,0.093252f,1.518149f,-0.933218f,-0.732996f,2.604095f,-0.118341f,-2.143904f,0.495076f,-2.179512f,-2.130751f,-0.996944f,0.261421f,-2.217463f,1.260067f,-0.150213f,3.790713f,1.481266f,-1.046058f,-1.523667f,-0.059621f,2.220780f,1.621212f,0.877230f,0.567247f,-3.175944f,-3.186688f,1.578665f};
static const float SHAPE_STD[32]={5.972266f,4.706852f,5.445010f,5.209927f,5.320220f,4.547237f,5.020802f,5.444004f,5.226681f,5.683095f,4.831436f,5.286469f,5.652043f,5.367606f,5.525084f,4.730578f,4.805265f,5.124013f,5.530808f,5.619001f,5.103930f,5.417670f,5.269677f,5.547194f,5.634698f,5.235274f,6.110351f,5.511298f,6.237273f,4.879207f,5.347008f,5.405691f};
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};

// SS decoder だけは backend を分ける。ggml WebGPU backend は GGML_OP_IM2COL_3D も
// GGML_OP_CONV_3D も実装していない（docs/PIXAL3D_WEBGPU_OP_GAP.md）。エラーにならず
// NaN 混じりの logits が返るため、SS の active voxel が潰れて以降の形状が全部壊れる
// （実測: browser の ss_logits が nonfinite=21143、ss_coords の bbox が x[2..4] に収縮）。
// trellis-test-pixal3d-ss-sample の dec_gpu=-1 と同じ扱いで CPU backend に載せる。
static constexpr int SS_DEC_BACKEND =
#ifdef __EMSCRIPTEN__
    -1;   // CPU
#else
    0;
#endif

struct I32A { vector<int64_t> shape; vector<int32_t> data; };
I32A load_i32(const string& path) {
    FILE* f=fopen(path.c_str(),"rb"); if(!f) throw std::runtime_error("cannot open "+path);
    unsigned char m[8]; if(fread(m,1,8,f)!=8 || memcmp(m,"\x93NUMPY",6)!=0){fclose(f);throw std::runtime_error("bad npy "+path);} uint16_t h=0; fread(&h,2,1,f);
    string hdr(h,'\0'); fread(hdr.data(),1,h,f); I32A a; size_t p=hdr.find("'shape':"); p=hdr.find('(',p); size_t q=hdr.find(')',p); string s=hdr.substr(p+1,q-p-1);
    for(size_t i=0;i<s.size();) if(isdigit((unsigned char)s[i])){int64_t v=0;while(i<s.size()&&isdigit((unsigned char)s[i]))v=v*10+(s[i++]-'0');a.shape.push_back(v);}else ++i;
    int64_t n=1; for(auto d:a.shape)n*=d; a.data.resize((size_t)n); if((int64_t)fread(a.data.data(),4,(size_t)n,f)!=n){fclose(f);throw std::runtime_error("short npy "+path);} fclose(f); return a;
}

vector<Pixal3dView> fixture_views(const string& dir, bool hi) {
    const string im = dir + (hi ? "/s1024_images.npy" : "/images_512.npy");
    npy::Array images=npy::load(im), fov=npy::load(dir+"/camera_angle_x.npy"), tm=npy::load(dir+"/transform_matrix.npy");
    const int V=(int)images.shape[1], S=(int)images.shape[3]; const size_t stride=(size_t)3*S*S;
    vector<Pixal3dView> v(V);
    for(int i=0;i<V;++i){v[i].rgb_premult.assign(images.data.begin()+i*stride,images.data.begin()+(i+1)*stride);v[i].fov_x=fov.data[i];memcpy(v[i].c2w,tm.data.data()+16*i,16*sizeof(float));}
    return v;
}

vector<float> deterministic_noise(const string& path, size_t n, uint32_t seed) {
    if(fexists(path)){auto a=npy::load(path); if(a.data.size()>=n) return vector<float>(a.data.begin(),a.data.begin()+n);} std::mt19937 r(seed); std::normal_distribution<float>d(0,1); vector<float>x(n);for(auto&z:x)z=d(r);return x;
}

vector<float> shape_flow(const string& gguf, const vector<array<int,3>>& coords, const Pixal3dCond& c, const vector<float>& noise) {
    Model m=Model::load(gguf,0); DiTParams p; p.in_ch=32;p.out_ch=32;p.d_cond=1024;
    if(!dit_detect_proj_attn(m,p)||p.d_proj!=2048) throw std::runtime_error("bad shape flow checkpoint");
    auto* r=make_sparse_runner(m,p,coords,c.n_global); vector<float> ng(c.global.size(),0), np(c.proj.size(),0);
    FlowFwdProj f=[&](const vector<float>&x,float ts,const float*cn,const float*pj){return r->forward(x,ts,cn,pj);};
    SamplerParams sp;sp.steps=12;sp.guidance_strength=7.5f;sp.guidance_rescale=.5f;sp.gi0=.6;sp.gi1=1.;sp.rescale_t=3.;
    auto o=sample_flow(f,noise,c.global.data(),ng.data(),c.proj.data(),np.data(),sp); delete r;m.free();return o;
}

vector<float> texture_flow(const string& gguf,const vector<array<int,3>>& coords,const Pixal3dCond& c,const vector<float>& shape_norm,const vector<float>& noise){
    const int N=(int)coords.size(); Model m=Model::load(gguf,0); DiTParams p;p.in_ch=64;p.out_ch=32;p.d_cond=1024;
    if(!dit_detect_proj_attn(m,p)||p.d_proj!=2048) throw std::runtime_error("bad texture flow checkpoint"); auto*r=make_sparse_runner(m,p,coords,c.n_global);
    vector<float> ng(c.global.size(),0),np(c.proj.size(),0),x64((size_t)64*N);
    FlowFwdProj f=[&](const vector<float>&st,float ts,const float*cn,const float*pj){for(int n=0;n<N;++n){for(int k=0;k<32;++k)x64[k+64*n]=st[k+32*n];for(int k=0;k<32;++k)x64[32+k+64*n]=shape_norm[k+32*n];}return r->forward(x64,ts,cn,pj);};
    SamplerParams sp;sp.steps=12;sp.guidance_strength=1.f;sp.guidance_rescale=0;sp.gi0=.6;sp.gi1=.9;sp.rescale_t=3.;
    auto o=sample_flow(f,noise,c.global.data(),ng.data(),c.proj.data(),np.data(),sp);delete r;m.free();return o;
}

int run_full_fixture(const vector<string>& model,const string& fixture,const string& out,uint32_t seed,bool tex_cond_fixture){
    // model order: dino, naf, ss_flow, ss_dec, shape512, shape_dec, shape1024, tex_flow, tex_dec
    if(model.size()!=9) return 2; frep("=== fixture-input full model E2E ===\n");
#ifdef __EMSCRIPTEN__
    g_no_fa=true;   // WebGPU backend に BF16 K/V FlashAttention は無い（--no-fa 相当の exact SDPA）
#endif
    auto v512=fixture_views(fixture,false),v1024=fixture_views(fixture,true); float mesh_scale=npy::load(fixture+"/mesh_scale.npy").data[0];
    // SS
    Pixal3dCond css; {Model d=Model::load(model[0],0);css=pixal3d_cond_ss_gpu(d,v512,512,16,mesh_scale);d.free();}
    vector<float> neg(css.global.size(),0),negp(css.proj.size(),0); Model sm=Model::load(model[2],0);DiTParams spm;spm.in_ch=8;spm.out_ch=8;spm.d_cond=1024;if(!dit_detect_proj_attn(sm,spm))return 3;auto*sr=make_dense_runner(sm,spm,16,css.n_global);
    FlowFwdProj sf=[&](const vector<float>&x,float ts,const float*c,const float*p){return sr->forward(x,ts,c,p);};SamplerParams ss;ss.steps=12;ss.guidance_strength=7.5f;ss.guidance_rescale=.7f;ss.gi0=.6;ss.gi1=1;ss.rescale_t=5;
    auto z=sample_flow(sf,deterministic_noise(fixture+"/noise.npy",8*4096,seed),css.global.data(),neg.data(),css.proj.data(),negp.data(),ss);delete sr;sm.free();vector<float>zdec(8*4096);for(int c=0;c<8;++c)for(int i=0;i<4096;++i)zdec[(size_t)c*4096+i]=z[c+8*i];
    vector<array<int,3>> coords;{Model d=Model::load(model[3],SS_DEC_BACKEND);auto logits=ss_decode(d,zdec);d.free();coords=ss_coords(logits,64,kSsRes);} if(coords.empty())return 4; frep("SS -> %zu coords\n",coords.size());
    // Shape512 live cond + flow
    Pixal3dCond c512;{Model d=Model::load(model[0],0),n=Model::load(model[1],0);Pixal3dSlatCondParams prm{512,kSsRes,512,mesh_scale};c512=pixal3d_cond_slat_gpu(d,n,v512,prm,nullptr,&coords);d.free();n.free();}auto lrnorm=shape_flow(model[4],coords,c512,deterministic_noise(fixture+"/shape_noise.npy",32*coords.size(),seed+1));vector<float>lrdn(lrnorm.size());for(size_t n=0;n<coords.size();++n)for(int c=0;c<32;++c)lrdn[c+32*n]=lrnorm[c+32*n]*SHAPE_STD[c]+SHAPE_MEAN[c];
    // upsample + Pixal3D quantize to grid64
    vector<array<int,3>> up;{Model d=Model::load(model[5],0);up=shape_upsample(d,lrdn,coords);d.free();}vector<array<int,3>>hr=pixal3d_quantize_hr_coords(up,kSsRes,1024);frep("Shape512 -> Shape1024 tokens=%zu\n",hr.size());
    // hr が確定したらもう要らない。
    up.clear();up.shrink_to_fit();
    // SS / Shape512 の conditioning と 512 の view はここから先で一度も読まない。
    css=Pixal3dCond{};c512=Pixal3dCond{};neg.clear();neg.shrink_to_fit();negp.clear();negp.shrink_to_fit();
    z.clear();z.shrink_to_fit();zdec.clear();zdec.shrink_to_fit();
    lrnorm.clear();lrnorm.shrink_to_fit();lrdn.clear();lrdn.shrink_to_fit();
    v512.clear();v512.shrink_to_fit();fheap("after Shape512");
    // Shape1024 live cond + flow (NAF T512, supported path)
    Pixal3dCond c1024;{Model d=Model::load(model[0],0),n=Model::load(model[1],0);Pixal3dSlatCondParams prm{1024,64,512,mesh_scale};c1024=pixal3d_cond_slat_gpu(d,n,v1024,prm,nullptr,&hr);d.free();n.free();}
    auto shnorm=shape_flow(model[6],hr,c1024,deterministic_noise(fixture+"/hr_shape_noise.npy",32*hr.size(),seed+2));vector<float>shdn(shnorm.size());for(size_t n=0;n<hr.size();++n)for(int c=0;c<32;++c)shdn[c+32*n]=shnorm[c+32*n]*SHAPE_STD[c]+SHAPE_MEAN[c];
    // c1024.proj は N*2048*4（N=17 660 で約 145 MB）。Shape1024 flow を抜けたら不要。
    c1024=Pixal3dCond{};fheap("after Shape1024");
    // Texture-1024 conditioning（live）。debug=1 のときだけ旧 fixture 経路。
    Pixal3dCond ct; bool live_tex_cond=true;
    if(tex_cond_fixture){
        live_tex_cond=false;
        ct.global=npy::load(fixture+"/tex_cond_global.npy").data;ct.n_global=5;ct.d_proj=2048;auto tp=npy::load(fixture+"/tex_cond_proj.npy");
        if(tp.shape.size()!=2 || tp.shape[0]!=(int64_t)hr.size()) throw std::runtime_error("fixture tex_cond_proj token count does not match live Shape1024 tokens; generate a matching fixture or use the live path");ct.proj=tp.data;
    } else {
        Model d=Model::load(model[0],0),n=Model::load(model[1],0);Pixal3dSlatCondParams prm{1024,64,1024,mesh_scale};Pixal3dCondStats cst;
        ct=pixal3d_cond_slat_gpu(d,n,v1024,prm,&cst,&hr);d.free();n.free();
        frep("live tex cond: tokens=%zu d_proj=%d graph peak %.1f MB resident %.1f MB %.1f s\n",hr.size(),ct.d_proj,cst.view_alloc_bytes/1048576.0,(cst.weight_bytes+cst.cond_bytes)/1048576.0,cst.total_ms/1000.0);
    }
    auto txnorm=texture_flow(model[7],hr,ct,shnorm,deterministic_noise(fixture+"/tex_noise.npy",32*hr.size(),seed+3));vector<float>txdn(txnorm.size());for(size_t n=0;n<hr.size();++n)for(int c=0;c<32;++c)txdn[c+32*n]=txnorm[c+32*n]*TEX_STD[c]+TEX_MEAN[c];
    // ct.proj も N*2048*4（約 145 MB）。Texture Flow を抜けたら conditioning は全部不要。
    ct=Pixal3dCond{};v1024.clear();v1024.shrink_to_fit();
    shnorm.clear();shnorm.shrink_to_fit();txnorm.clear();txnorm.shrink_to_fit();fheap("after TextureFlow");
    ShapeOut so;{Model d=Model::load(model[5],0);so=shape_decode(d,shdn,hr,1024);d.free();}
    shdn.clear();shdn.shrink_to_fit();
    frep("shape_decode done: %zu voxels res=%d\n",so.coords.size(),so.res);fheap("shape_decode");
    Mesh mesh=dual_grid_to_mesh(so);if(mesh.F()<=0)return 5;
    // feats7 は [7,M]（M=4.76M で約 133 MB）。メッシュを作ったら tex decoder には要らない。
    so.feats7.clear();so.feats7.shrink_to_fit();
    frep("mesh V=%d F=%d\n",mesh.V(),mesh.F());fheap("dual_grid_to_mesh");
    vector<float>raw;{Model d=Model::load(model[8],0);raw=tex_decode(d,txdn,hr,so.subs);d.free();}if(raw.size()!=so.coords.size()*6)return 5;
    frep("tex_decode done: %zu voxels x6 (%.1f MB)\n",so.coords.size(),raw.size()*4/1048576.0);fheap("tex_decode");
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
    if(!pixal3d_write_production_glb(out,mesh,so.coords,pbr,so.res,opt,&pr)){frep("%s",pr.c_str());return 5;}frep("%s",pr.c_str());
    frep("FULL_E2E_LIVE_SS=1\nFULL_E2E_LIVE_SHAPE512=1\nFULL_E2E_LIVE_SHAPE1024=1\nFULL_E2E_LIVE_TEXTURE_COND=%d\nFULL_E2E_TEXTURE_FLOW=1\nFULL_E2E_GLB=1\n",live_tex_cond?1:0);
    frep("FULL_MODEL_E2E_RESULT: %s V=%d F=%d\n",live_tex_cond?"LIVE_ALL_STAGES":"PARTIAL_LIVE_TEXTURE_COND",mesh.V(),mesh.F());return 0;
}
}

extern "C" PIXAL3D_FULL_EXPORT const char* pixal3d_full_fixture_run(const char* dinov3,const char* naf,const char* ss_flow,const char* ss_dec,const char* shape512,const char* shape_dec,const char* shape1024,const char* tex_flow,const char* tex_dec,const char* fixture,const char* out_glb,int seed,int tex_cond_fixture){
    g_full_report.clear();try{vector<string>m={dinov3,naf,ss_flow,ss_dec,shape512,shape_dec,shape1024,tex_flow,tex_dec};int rc=run_full_fixture(m,fixture?fixture:"",out_glb?out_glb:"/out/full_fixture.glb",seed?seed:1,tex_cond_fixture!=0);frep(rc==0?(tex_cond_fixture?"FULL_FIXTURE_RESULT: OK_WITH_TEXTURE_COND_FIXTURE\n":"FULL_FIXTURE_RESULT: OK_LIVE_TEXTURE_COND\n"):"FULL_FIXTURE_RESULT: FAIL rc=%d\n",rc);}catch(const std::exception&e){frep("FULL_FIXTURE_RESULT: EXCEPTION %s\n",e.what());}return g_full_report.c_str();}
