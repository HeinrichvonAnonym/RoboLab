#include "camera/rgbd.h"

#include <librealsense2/rsutil.h>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace robo_lab::camera_support {
namespace {
template<class T> void read(const YAML::Node& n, const char* key, T& value) {
  if (n[key]) value = n[key].as<T>();
}
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool finite(float x) { return std::isfinite(x); }
rs2_intrinsics intrinsics(const camera::Intrinsics& in) {
  rs2_intrinsics out{};
  out.width = static_cast<int>(in.width()); out.height = static_cast<int>(in.height());
  out.fx = in.fx(); out.fy = in.fy(); out.ppx = in.ppx(); out.ppy = in.ppy();
  out.model = static_cast<rs2_distortion>(in.distortion_model());
  std::copy(in.coefficients().begin(), in.coefficients().end(), out.coeffs);
  return out;
}
void validate_intrinsics(const camera::Intrinsics& k, const camera::Image& image) {
  require(k.width() == image.width() && k.height() == image.height(), "intrinsics/image dimensions disagree");
  require(finite(k.fx()) && finite(k.fy()) && k.fx() > 0 && k.fy() > 0 && finite(k.ppx()) && finite(k.ppy()), "invalid focal length/principal point");
  require(k.distortion_model() < RS2_DISTORTION_COUNT && k.coefficients_size() == 5, "invalid distortion calibration");
  for (float c : k.coefficients()) require(finite(c), "non-finite distortion coefficient");
}
void validate_image(const camera::Image& image, camera::Image::Encoding encoding, uint32_t bytes) {
  require(image.encoding() == encoding, "unexpected image encoding");
  require(image.width() > 0 && image.height() > 0 && image.width() <= 8192 && image.height() <= 8192, "invalid image dimensions");
  require(image.stride() >= uint64_t(image.width()) * bytes && image.stride() <= uint64_t(image.width()) * bytes + 65536, "invalid image stride");
  require(uint64_t(image.stride()) * image.height() == image.data().size(), "image payload size disagrees with stride");
  require(std::isfinite(image.device_timestamp_ms()) && image.device_timestamp_ms() >= 0 && image.timestamp_domain() <= 2, "invalid image timestamp");
}
}  // namespace

CaptureConfig CaptureConfig::load(const YAML::Node& n) {
  require(n.IsMap(), "camera configuration must be a map");
  CaptureConfig c;
  read(n,"serial",c.serial); read(n,"width",c.width); read(n,"height",c.height);
  // Previously fps was stored as float. Keep integral YAML values such as 15.0 valid.
  if (n["fps"]) {
    const double f = n["fps"].as<double>();
    require(std::isfinite(f) && f >= 1 && f <= 90 && std::floor(f) == f, "fps must be an integer from 1 to 90");
    c.fps = static_cast<int>(f);
  }
  read(n,"enable_rgb",c.rgb); read(n,"enable_depth",c.depth); read(n,"enable_ir",c.ir);
  read(n,"publish_rgbd",c.publish_rgbd); read(n,"use_opengl",c.use_opengl);
  read(n,"max_pair_skew_ms",c.max_pair_skew_ms);
  if (n["topic"]) {
    require(n["topic"].IsMap(), "topic must be a map");
    read(n["topic"],"rgb",c.rgb_topic); read(n["topic"],"depth",c.depth_topic);
    read(n["topic"],"ir",c.ir_topic); read(n["topic"],"rgbd",c.rgbd_topic);
  }
  require(c.width > 0 && c.height > 0 && c.width <= 8192 && c.height <= 8192, "invalid stream dimensions");
  require(c.rgb || c.depth || c.ir, "all streams are disabled");
  require(std::isfinite(c.max_pair_skew_ms) && c.max_pair_skew_ms >= 0, "invalid maximum pair skew");
  require(!c.publish_rgbd || (c.rgb && c.depth && !c.rgbd_topic.empty()), "RGB-D publishing requires color, depth, and a topic");
  return c;
}

bool PairGate::accept(const camera::Image& c, const camera::Image& d, double limit) {
  if (c.data().empty() || d.data().empty() || !std::isfinite(limit) || limit < 0 ||
      !std::isfinite(c.device_timestamp_ms()) || !std::isfinite(d.device_timestamp_ms()) ||
      c.timestamp_domain() != d.timestamp_domain() || c.timestamp_domain() > 2 ||
      std::abs(c.device_timestamp_ms() - d.device_timestamp_ms()) > limit ||
      (have_previous && (c.frame_number() <= last_color || d.frame_number() <= last_depth))) return false;
  last_color = c.frame_number(); last_depth = d.frame_number(); have_previous = true;
  return true;
}

void validate_frame(const camera::RgbdFrame& f) {
  require(f.schema_version() == 1, "unsupported RGB-D schema version");
  validate_image(f.color(), camera::Image::RGB8, 3);
  validate_image(f.depth(), camera::Image::Z16, 2);
  validate_intrinsics(f.color_intrinsics(), f.color());
  validate_intrinsics(f.depth_intrinsics(), f.depth());
  // rsutil does not support deprojection of the forward-only modified Brown model.
  require(f.depth_intrinsics().distortion_model() != RS2_DISTORTION_MODIFIED_BROWN_CONRADY, "unsupported depth deprojection model");
  require(finite(f.depth_units_m()) && f.depth_units_m() > 0, "invalid depth units");
  require(f.depth_to_color().rotation_size() == 9 && f.depth_to_color().translation_m_size() == 3, "invalid extrinsics dimensions");
  for (float x : f.depth_to_color().rotation()) require(finite(x), "non-finite rotation");
  for (float x : f.depth_to_color().translation_m()) require(finite(x), "non-finite translation");
  Eigen::Map<const Eigen::Matrix3f> rotation(f.depth_to_color().rotation().data());
  require((rotation.transpose()*rotation - Eigen::Matrix3f::Identity()).norm() < 0.01f && rotation.determinant() > 0.99f, "extrinsics rotation is not orthonormal");
}

PreviewConfig PreviewConfig::load(const YAML::Node& n) {
  require(n.IsMap(), "preview configuration must be a map");
  PreviewConfig c;
  read(n,"topic",c.topic); read(n,"min_depth_m",c.min_depth_m); read(n,"max_depth_m",c.max_depth_m);
  read(n,"eye_separation_m",c.eye_separation_m); read(n,"convergence_m",c.convergence_m);
  read(n,"max_depth_jump_m",c.max_depth_jump_m); c.validate(); return c;
}
void PreviewConfig::validate() const {
  require(!topic.empty(), "empty RGB-D topic");
  require(finite(min_depth_m) && finite(max_depth_m) && min_depth_m > 0 && max_depth_m > min_depth_m, "require 0 < min_depth_m < max_depth_m");
  require(finite(eye_separation_m) && eye_separation_m >= 0 && eye_separation_m <= 0.20f, "eye_separation_m must be between 0 and 0.20 metres");
  require(finite(convergence_m) && convergence_m > 0, "convergence_m must be positive");
  require(finite(max_depth_jump_m) && max_depth_jump_m > 0, "max_depth_jump_m must be positive");
}

Mesh build_mesh(const camera::RgbdFrame& f, const PreviewConfig& c) {
  validate_frame(f); c.validate();
  const auto dk = intrinsics(f.depth_intrinsics()), ck = intrinsics(f.color_intrinsics());
  const auto& image = f.depth(); const uint32_t w = image.width(), h = image.height();
  Mesh mesh; mesh.total_depth_pixels = size_t(w)*h; mesh.vertices.resize(mesh.total_depth_pixels);
  std::vector<float> depth(mesh.total_depth_pixels, 0);
  Eigen::Map<const Eigen::Matrix3f> rotation(f.depth_to_color().rotation().data());
  Eigen::Map<const Eigen::Vector3f> translation(f.depth_to_color().translation_m().data());
  for (uint32_t y=0; y<h; ++y) for (uint32_t x=0; x<w; ++x) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(image.data().data()) + size_t(y)*image.stride()+2*x;
    const float z = float(uint16_t(bytes[0]) | (uint16_t(bytes[1])<<8))*f.depth_units_m();
    if (z < c.min_depth_m || z > c.max_depth_m) continue;
    ++mesh.valid_depth_pixels;
    float pixel[2]{float(x),float(y)}, point[3];
    rs2_deproject_pixel_to_point(point,&dk,pixel,z);
    auto& vertex=mesh.vertices[size_t(y)*w+x];
    vertex.position=rotation*Eigen::Map<Eigen::Vector3f>(point)+translation;
    if (!vertex.position.allFinite() || vertex.position.z() <= 0) continue;
    rs2_project_point_to_pixel(pixel,&ck,vertex.position.data());
    if (!std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) || pixel[0]<0 || pixel[1]<0 || pixel[0]>ck.width-1 || pixel[1]>ck.height-1) continue;
    vertex.uv=Eigen::Vector2f((pixel[0]+0.5f)/ck.width,(pixel[1]+0.5f)/ck.height);
    depth[size_t(y)*w+x]=z;
  }
  mesh.indices.reserve(size_t(w-1)*(h-1)*6);
  auto triangle=[&](uint32_t a,uint32_t b,uint32_t d) {
    if (depth[a] == 0 || depth[b] == 0 || depth[d] == 0) return;
    const auto bounds=std::minmax({depth[a],depth[b],depth[d]});
    if (bounds.second-bounds.first <= c.max_depth_jump_m) mesh.indices.insert(mesh.indices.end(),{a,b,d});
  };
  for(uint32_t y=0;y+1<h;++y) for(uint32_t x=0;x+1<w;++x) {
    const uint32_t a=y*w+x; triangle(a,a+w,a+1); triangle(a+1,a+w,a+w+1);
  }
  return mesh;
}

Eigen::Matrix4f eye_matrix(const camera::Intrinsics& k,const PreviewConfig& c,float eye_x) {
  c.validate();
  // Pixel centre convention; source +Y is down, OpenGL NDC +Y is up.
  const float sx=2*k.fx()/k.width(), sy=2*k.fy()/k.height();
  const float near=c.min_depth_m*0.5f, far=c.max_depth_m*2;
  Eigen::Matrix4f m=Eigen::Matrix4f::Zero();
  m(0,0)=sx; m(0,2)=2*(k.ppx()+0.5f)/k.width()-1+sx*eye_x/c.convergence_m; m(0,3)=-sx*eye_x;
  m(1,1)=-sy; m(1,2)=1-2*(k.ppy()+0.5f)/k.height();
  m(2,2)=(far+near)/(far-near); m(2,3)=-2*far*near/(far-near); m(3,2)=1;
  return m;
}
}  // namespace robo_lab::camera_support
