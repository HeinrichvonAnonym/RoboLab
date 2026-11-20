#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include "camera/rgbd_renderer.h"
#include "rgbd_fixture.h"
#include <gtest/gtest.h>
#include <fstream>

using namespace robo_lab::camera_support;
TEST(StereoRender, OcclusionZeroEyeAndMissingDepth) {
  ASSERT_TRUE(glfwInit()); glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
  auto* window=glfwCreateWindow(128,128,"RGBD render test",nullptr,nullptr); ASSERT_NE(window,nullptr);
  glfwMakeContextCurrent(window);
  {
    RgbdRenderer renderer; auto f=rgbd_fixture(64,64,1); PreviewConfig config;
    Mesh mesh;
    // Near red triangle, then an overlapping far blue triangle: the later draw must not win.
    mesh.vertices={{{-.08f,-.08f,.2f},{.25f,.5f}},{{.08f,-.08f,.2f},{.25f,.5f}},{{0,.08f,.2f},{.25f,.5f}},
                   {{-.16f,-.16f,.4f},{.75f,.5f}},{{.16f,-.16f,.4f},{.75f,.5f}},{{0,.16f,.4f},{.75f,.5f}}};
    mesh.indices={0,1,2,3,4,5};renderer.upload(f,mesh);
    auto draw=[&](float eye) {
      glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
      renderer.draw_eye(eye_matrix(f.color_intrinsics(),config,eye),0,0,128,128);
      std::vector<unsigned char> pixels(128*128*3);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,128,128,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());return pixels;
    };
    const auto left=draw(0),right=draw(0);EXPECT_EQ(left,right);
    const int centre=(64*128+64)*3;EXPECT_GT(left[centre],240);EXPECT_LT(left[centre+2],10);
    EXPECT_EQ(left[0],0);EXPECT_EQ(left[1],0);EXPECT_EQ(left[2],0);
    if(const char* output=std::getenv("RGBD_QA_IMAGE")) {
      std::ofstream file(output,std::ios::binary);file<<"P6\n128 128\n255\n";
      for(int y=127;y>=0;--y)file.write(reinterpret_cast<const char*>(left.data()+y*128*3),128*3);
    }
    mesh.indices.clear();renderer.upload(f,mesh);const auto empty=draw(0);
    EXPECT_TRUE(std::all_of(empty.begin(),empty.end(),[](auto x){return x==0;}));
    EXPECT_EQ(glGetError(),GL_NO_ERROR);
  }
  glfwDestroyWindow(window);glfwTerminate();
}
