// Fixture-injected browser E2E:
// Shape-1024 fixture -> Texture Flow -> Shape Decode -> Texture Decode -> production postprocess -> GLB.
#define PIXAL3D_NO_NATIVE_MAIN   // includee の native main を抑止（下に自前の main がある）
#include "pixal3d_texture_wasm.cpp"
#include "shape_decoder.h"
#include "dual_grid.h"
#include "pixal3d_postprocess.h"
#include "trellis_model.h"
#include "npy.h"
#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
using std::array; using std::string; using std::vector; using namespace trellis;
namespace {
static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};
int run_partial(const string& tex_flow,const string& shape_dec,const string& tex_dec,const string& fixture,const string& concat,const string& out,int resolution){
    rep("=== partial E2E + production postprocess ===\n");
    int rc=run_texture_impl("","",tex_flow,"",fixture,0,0,concat); if(rc) return rc; const vector<float> txnorm=g_latent;
    vector<int32_t> co; vector<int64_t> cshape; if(!load_npy_i32(fixture+"/hr_coords.npy",co,cshape)||cshape.size()!=2)return 3; const int64_t N=cshape[0],cw=cshape[1];
    vector<array<int,3>> coords((size_t)N); int cmax=0; for(int64_t i=0;i<N;++i){coords[i]={co[i*cw+cw-3],co[i*cw+cw-2],co[i*cw+cw-1]};cmax=std::max({cmax,coords[i][0],coords[i][1],coords[i][2]});}
    const int res=resolution>0?resolution:(cmax<32?512:1024); string shp=fixture+"/f32_shape_slat.npy";if(!file_exists(shp))shp=fixture+"/f32_slat.npy";if(!file_exists(shp))return 3;auto sn=npy::load(shp);if(sn.numel()!=N*32)return 3;vector<float> sh(sn.data.begin(),sn.data.end());
    vector<float> mean(TEX_MEAN,TEX_MEAN+32),stdv(TEX_STD,TEX_STD+32);if(file_exists(fixture+"/tex_norm_mean.npy"))mean=npy::load(fixture+"/tex_norm_mean.npy").data;if(file_exists(fixture+"/tex_norm_std.npy"))stdv=npy::load(fixture+"/tex_norm_std.npy").data;
    vector<float> tx((size_t)N*32);for(int64_t n=0;n<N;++n)for(int c=0;c<32;++c)tx[c+32*n]=txnorm[c+32*n]*(c<(int)stdv.size()?stdv[c]:TEX_STD[c])+(c<(int)mean.size()?mean[c]:TEX_MEAN[c]);
    ShapeOut so;{Model m=Model::load(shape_dec,0);so=shape_decode(m,sh,coords,res);m.free();}Mesh mesh=dual_grid_to_mesh(so);if(mesh.F()<=0)return 4;vector<float> raw;{Model m=Model::load(tex_dec,0);raw=tex_decode(m,tx,coords,so.subs);m.free();}if(raw.size()!=so.coords.size()*6)return 4;
    vector<float> pbr(raw.size());for(size_t i=0;i<raw.size();++i)pbr[i]=std::clamp(.5f*raw[i]+.5f,0.f,1.f);
    Pixal3dPostprocessOptions opt;opt.remesh_band=1;opt.use_xatlas=false;opt.use_webp=false;opt.texture_size=4096;
#ifdef __EMSCRIPTEN__
 // wasm32 のヒープは 4 GiB が上限。res=1024 の narrow-band remesh は実測 7.8M 頂点 /
 // 15.6M 面を作り、その後の QEM と合わせて収まらない（std::bad_alloc）。粗いグリッドで
 // 同じ経路を回す。使った値は postprocess の report 行に出る。
 // 実測（2026-09-09, cyclops: decode 9 485 680 面）:
 //   512  -> remesh 4 536 496 面、host live ピーク 1 869 MB、完走
 //   1024 -> remesh に入った時点で live 1 644 MB、18 318 052 面を作ろうとして std::bad_alloc
 // native の最大 RSS は 512 で 5.39 GB / 1024 で 7.30 GB。wasm の live ピークは native の
 // 約 1/3（ポインタ幅が半分・vector の容量確保の差）だが、それでも 1024 は 4 GiB に入らない。
 opt.remesh_res=512;opt.target_faces=500000;
#else
 opt.target_faces=1000000;
#endif
    // 予算の A/B のための上書き（既定はこの上で決まる）。ブラウザからは worker が
    // Module.ENV に入れた値がここに届く。
    if(const char*e=getenv("PIXAL3D_REMESH_RES"))opt.remesh_res=atoi(e);
    if(const char*e=getenv("PIXAL3D_TARGET_FACES"))opt.target_faces=atoi(e);
    string pr;bool ok=pixal3d_write_production_glb(out,mesh,so.coords,pbr,so.res,opt,&pr);rep("%s",pr.c_str());if(!ok)return 5;
    FILE*f=fopen(out.c_str(),"rb");long bytes=-1;if(f){fseek(f,0,SEEK_END);bytes=ftell(f);fclose(f);}rep("partial-e2e: textured GLB path=%s bytes=%ld atlas=%d\n",out.c_str(),bytes,opt.texture_size);return 0;
}
}
extern "C" PIXAL3D_EXPORT const char* pixal3d_partial_e2e_run(const char* tex_flow,const char* shape_dec,const char* tex_dec,const char* fixture,const char* concat,const char* out,int resolution){g_report.clear();int rc=1;try{rc=run_partial(tex_flow,shape_dec,tex_dec,fixture?fixture:"",concat?concat:"",out?out:"/out/partial_e2e.glb",resolution);}catch(const std::exception&e){rep("partial-e2e EXCEPTION: %s\n",e.what());rc=1;}rep(rc==0?"PARTIAL_E2E_RESULT: OK\n":"PARTIAL_E2E_RESULT: FAIL rc=%d\n",rc);return g_report.c_str();}


#ifndef __EMSCRIPTEN__
// native ドライバ。ブラウザを起動せずに partial 経路（Texture Flow -> shape decode ->
// tex decode -> production postprocess -> textured GLB）を同じ C++ で通すためのもの。
// 重みの量子化や分割の A/B を native で完結させられる。
//   pixal3d-partial-run <tex_flow.gguf> <shape_dec.gguf> <tex_dec.gguf> <fixture_dir>
//                       <out.glb> [concat_cond.npy] [resolution]
int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <tex_flow.gguf> <shape_dec.gguf> <tex_dec.gguf> "
                        "<fixture_dir> <out.glb> [concat_cond.npy] [resolution]\n", argv[0]);
        return 2;
    }
    const char* concat = argc > 6 ? argv[6] : "";
    const int res = argc > 7 ? atoi(argv[7]) : 0;
    const char* r = pixal3d_partial_e2e_run(argv[1], argv[2], argv[3], argv[4], concat, argv[5], res);
    const bool ok = strstr(r, "PARTIAL_E2E_RESULT: OK") != nullptr;
    printf("\n%s\n", ok ? "PARTIAL_E2E: PASS" : "PARTIAL_E2E: FAIL");
    return ok ? 0 : 1;
}
#endif
