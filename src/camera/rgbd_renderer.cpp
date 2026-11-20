#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include "camera/rgbd_renderer.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace robo_lab::camera_support {
namespace {
GLuint shader(GLenum kind,const char* source) {
  GLuint id=glCreateShader(kind); glShaderSource(id,1,&source,nullptr); glCompileShader(id);
  GLint ok; glGetShaderiv(id,GL_COMPILE_STATUS,&ok);
  if(!ok) { char log[2048]; glGetShaderInfoLog(id,sizeof(log),nullptr,log); glDeleteShader(id); throw std::runtime_error(log); }
  return id;
}
void attributes() {
  glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,position)));
  glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,uv)));
}
}  // namespace
struct RgbdRenderer::Impl {
  GLuint program{},texture{},vao{},vbo{},ebo{},quad_vao{},quad_vbo{};
  GLsizei count{};
  GLint matrix_location{};
  ~Impl() {
    glDeleteProgram(program); glDeleteTextures(1,&texture);
    glDeleteBuffers(1,&vbo); glDeleteBuffers(1,&ebo); glDeleteBuffers(1,&quad_vbo);
    glDeleteVertexArrays(1,&vao); glDeleteVertexArrays(1,&quad_vao);
  }
  void use(const Eigen::Matrix4f& matrix,int x,int y,int w,int h) {
    glViewport(x,y,w,h); glUseProgram(program);
    glUniformMatrix4fv(matrix_location,1,GL_FALSE,matrix.data());
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,texture);
  }
};
RgbdRenderer::RgbdRenderer():impl_(std::make_unique<Impl>()) {
  auto& p=*impl_;
  const char* vs=R"(#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec2 texcoord;
uniform mat4 matrix;
out vec2 uv;
void main(){ gl_Position=matrix*vec4(position,1.0); uv=texcoord; })";
  const char* fs=R"(#version 330 core
in vec2 uv;
uniform sampler2D color_image;
out vec4 color;
void main(){ color=vec4(texture(color_image,uv).rgb,1.0); })";
  GLuint v=shader(GL_VERTEX_SHADER,vs), f=shader(GL_FRAGMENT_SHADER,fs);
  p.program=glCreateProgram(); glAttachShader(p.program,v); glAttachShader(p.program,f); glLinkProgram(p.program);
  glDeleteShader(v); glDeleteShader(f);
  GLint ok; glGetProgramiv(p.program,GL_LINK_STATUS,&ok);
  if(!ok) throw std::runtime_error("RGB-D shader linking failed");
  p.matrix_location=glGetUniformLocation(p.program,"matrix");
  glUseProgram(p.program); glUniform1i(glGetUniformLocation(p.program,"color_image"),0);
  glGenTextures(1,&p.texture); glBindTexture(GL_TEXTURE_2D,p.texture);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
  glGenVertexArrays(1,&p.vao); glBindVertexArray(p.vao);
  glGenBuffers(1,&p.vbo); glBindBuffer(GL_ARRAY_BUFFER,p.vbo); attributes();
  glGenBuffers(1,&p.ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,p.ebo);
  const Vertex quad[]={{{-1,-1,0},{0,1}},{{1,-1,0},{1,1}},{{1,1,0},{1,0}},{{-1,-1,0},{0,1}},{{1,1,0},{1,0}},{{-1,1,0},{0,0}}};
  glGenVertexArrays(1,&p.quad_vao); glBindVertexArray(p.quad_vao);
  glGenBuffers(1,&p.quad_vbo); glBindBuffer(GL_ARRAY_BUFFER,p.quad_vbo);
  glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW); attributes();
  glDisable(GL_CULL_FACE);
}
RgbdRenderer::~RgbdRenderer()=default;
void RgbdRenderer::upload(const camera::RgbdFrame& frame,const Mesh& mesh) {
  const auto& c=frame.color(); auto& p=*impl_;
  std::vector<char> packed; const char* pixels=c.data().data();
  if(c.stride()!=c.width()*3) {
    packed.resize(size_t(c.width())*c.height()*3);
    for(uint32_t y=0;y<c.height();++y) std::memcpy(packed.data()+size_t(y)*c.width()*3,pixels+size_t(y)*c.stride(),c.width()*3);
    pixels=packed.data();
  }
  glBindTexture(GL_TEXTURE_2D,p.texture); glPixelStorei(GL_UNPACK_ALIGNMENT,1);
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGB8,c.width(),c.height(),0,GL_RGB,GL_UNSIGNED_BYTE,pixels);
  glBindVertexArray(p.vao); glBindBuffer(GL_ARRAY_BUFFER,p.vbo);
  glBufferData(GL_ARRAY_BUFFER,mesh.vertices.size()*sizeof(Vertex),mesh.vertices.data(),GL_STREAM_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,p.ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,mesh.indices.size()*sizeof(uint32_t),mesh.indices.data(),GL_STREAM_DRAW);
  p.count=static_cast<GLsizei>(mesh.indices.size());
}
void RgbdRenderer::draw_original(int x,int y,int w,int h) {
  auto& p=*impl_; p.use(Eigen::Matrix4f::Identity(),x,y,w,h);
  glDisable(GL_DEPTH_TEST); glBindVertexArray(p.quad_vao); glDrawArrays(GL_TRIANGLES,0,6);
}
void RgbdRenderer::draw_eye(const Eigen::Matrix4f& matrix,int x,int y,int w,int h) {
  auto& p=*impl_; p.use(matrix,x,y,w,h);
  glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glBindVertexArray(p.vao); glDrawElements(GL_TRIANGLES,p.count,GL_UNSIGNED_INT,nullptr);
}
}  // namespace robo_lab::camera_support
