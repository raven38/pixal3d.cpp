#include "pixal3d_input.h"
#include "transforms_json.h"
#include "stb_image.h"
#include "stb_image_resize.h"
#include <algorithm>
#include <cstring>

namespace trellis {
namespace {
std::vector<unsigned char> load_rgba(const std::string& path,int& W,int& H,bool& alpha){int ch=0;unsigned char* p=stbi_load(path.c_str(),&W,&H,&ch,4);if(!p){alpha=false;return{};}alpha=false;for(size_t i=0;i<(size_t)W*H;++i)if(p[4*i+3]<250){alpha=true;break;}std::vector<unsigned char>o(p,p+(size_t)W*H*4);stbi_image_free(p);return o;}
std::vector<float> resize_premult(const std::vector<unsigned char>& rgba,int W,int H,int S){std::vector<unsigned char>rs((size_t)S*S*4);stbir_resize_uint8(rgba.data(),W,H,0,rs.data(),S,S,0,4);std::vector<float>o((size_t)3*S*S);for(int y=0;y<S;++y)for(int x=0;x<S;++x){size_t i=((size_t)y*S+x)*4;float a=rs[i+3]/255.f;for(int c=0;c<3;++c)o[(size_t)c*S*S+(size_t)y*S+x]=(rs[i+c]/255.f)*a;}return o;}
}
bool pixal3d_load_input_views(const std::string& dir,Pixal3dInputViews& out,std::string& error,int max_views){TransformsFile tf;if(!load_transforms_json(dir+"/transforms.json",tf)){error="failed to load transforms.json";return false;}const int n=max_views>0?std::min<int>(max_views,(int)tf.frames.size()):(int)tf.frames.size();if(n<=0){error="transforms.json has no frames";return false;}out.views512.clear();out.views1024.clear();out.views512.reserve(n);out.views1024.reserve(n);out.mesh_scale=tf.mesh_scale;out.has_mesh_scale=tf.has_mesh_scale;for(int i=0;i<n;++i){const auto&fr=tf.frames[i];float fov=fr.has_camera_angle_x?fr.camera_angle_x:(tf.has_camera_angle_x?tf.camera_angle_x:0.f);if(fov<=0){error="frame "+std::to_string(i)+" has no camera_angle_x";return false;}int W=0,H=0;bool alpha=false;auto rgba=load_rgba(dir+"/"+fr.file_path,W,H,alpha);if(rgba.empty()){error="cannot load "+fr.file_path;return false;}if(!alpha){error="frame "+fr.file_path+" has no real alpha channel; pre-matted RGBA is required";return false;}Pixal3dView a{},b{};a.rgb_premult=resize_premult(rgba,W,H,512);b.rgb_premult=resize_premult(rgba,W,H,1024);a.fov_x=b.fov_x=fov;memcpy(a.c2w,fr.transform_matrix,16*sizeof(float));memcpy(b.c2w,fr.transform_matrix,16*sizeof(float));out.views512.push_back(std::move(a));out.views1024.push_back(std::move(b));}return true;}
}
