#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include "camera/rgbd.h"
#include "camera/rgbd_renderer.h"
#include "message_system.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

namespace {
volatile std::sig_atomic_t interrupted=0;
void interrupt(int) { interrupted=1; }
using Clock=std::chrono::steady_clock;
struct Mailbox {
  std::mutex mutex;
  std::string payload;
  Clock::time_point received{};
  uint64_t received_count{},replaced{};
};
struct Glfw {
  GLFWwindow* window{};
  ~Glfw() { if(window) glfwDestroyWindow(window); glfwTerminate(); }
};
}

int main(int argc,char** argv) {
  using namespace robo_lab::camera_support;
  try {
    std::string path="config/rgbd_stereo_preview.yaml",snapshot;
    bool headless=false,hidden=false; double duration=0;
    for(int i=1;i<argc;++i) {
      std::string arg=argv[i];
      if(arg=="--headless") headless=true;
      else if(arg=="--hidden") hidden=true;
      else if(arg=="--snapshot" && i+1<argc) snapshot=argv[++i];
      else if(arg=="--duration" && i+1<argc) { duration=std::stod(argv[++i]); if(!std::isfinite(duration)||duration<=0) throw std::invalid_argument("duration must be positive"); }
      else if(arg=="--help") { std::cout << "rgbd_stereo_preview [config.yaml] [--headless|--hidden] [--duration SECONDS] [--snapshot image.ppm]\nPanels: original RGB | synthetic left | synthetic right. Escape closes.\n"; return 0; }
      else if(arg.rfind("--",0)==0) throw std::invalid_argument("unknown/incomplete option: "+arg);
      else path=arg;
    }
    if(headless && !snapshot.empty()) throw std::invalid_argument("--snapshot requires a GL display (can use --hidden)");
    auto config=PreviewConfig::load(YAML::LoadFile(path));
    std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
    Glfw glfw;
    std::unique_ptr<RgbdRenderer> renderer;
    if(!headless) {
      glfwSetErrorCallback([](int,const char* text){std::cerr << "GLFW: " << text << '\n';});
      if(!glfwInit()) throw std::runtime_error("cannot initialize display; use --headless for geometry/stats only");
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
      glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_VISIBLE,hidden?GLFW_FALSE:GLFW_TRUE);
      glfw.window=glfwCreateWindow(1536,384,"D405: original RGB | synthetic LEFT | synthetic RIGHT",nullptr,nullptr);
      if(!glfw.window) throw std::runtime_error("OpenGL 3.3 window creation failed");
      glfwMakeContextCurrent(glfw.window); glfwSwapInterval(hidden?0:1);
      renderer=std::make_unique<RgbdRenderer>();
      std::cout << "renderer=" << glGetString(GL_RENDERER) << std::endl;
    }
    // The callback only replaces one pending payload. Mesh work stays off Zenoh threads.
    auto mailbox=std::make_shared<Mailbox>();
    robo_lab::MessageSystem messages; messages.initialize();
    if(!messages.is_open() || !messages.subscribe(config.topic,[mailbox](const std::string&,const std::string& payload) {
      if(payload.size()>64*1024*1024) return;
      std::lock_guard<std::mutex> lock(mailbox->mutex);
      if(!mailbox->payload.empty()) ++mailbox->replaced;
      mailbox->payload=payload; mailbox->received=Clock::now(); ++mailbox->received_count;
    })) throw std::runtime_error("could not subscribe to RGB-D topic");
    const auto start=Clock::now(); auto report=start, last_received=start;
    uint64_t processed=0,invalid=0,previous_count=0,frame_gaps=0,last_frame=0;
    double max_skew=0, valid_ratio=0, max_age_ms=0;
    camera::RgbdFrame current;
    bool have_frame=false,was_stale=false;
    while(!interrupted && (!glfw.window || !glfwWindowShouldClose(glfw.window))) {
      const auto now=Clock::now();
      if(duration>0 && std::chrono::duration<double>(now-start).count()>=duration) break;
      std::string payload; Clock::time_point arrival;
      { std::lock_guard<std::mutex> lock(mailbox->mutex); payload.swap(mailbox->payload); arrival=mailbox->received; }
      if(!payload.empty()) {
        try {
          camera::RgbdFrame next;
          if(!next.ParseFromString(payload)) throw std::invalid_argument("invalid RGB-D protobuf");
          auto mesh=build_mesh(next,config);
          if(renderer) renderer->upload(next,mesh);
          if(last_frame && next.color().frame_number()>last_frame+1) frame_gaps+=next.color().frame_number()-last_frame-1;
          last_frame=next.color().frame_number();
          max_skew=std::max(max_skew,std::abs(next.color().device_timestamp_ms()-next.depth().device_timestamp_ms()));
          max_age_ms=std::max(max_age_ms,std::chrono::duration<double,std::milli>(Clock::now().time_since_epoch()).count()-next.host_received_monotonic_ns()/1e6);
          valid_ratio=double(mesh.valid_depth_pixels)/mesh.total_depth_pixels;
          current.Swap(&next); ++processed; have_frame=true; last_received=arrival;
        } catch(const std::exception& e) { ++invalid; if(invalid<=5) std::cerr << "rejected frame: " << e.what() << '\n'; }
      }
      const bool stale=!have_frame || Clock::now()-last_received>std::chrono::milliseconds(500);
      if(stale!=was_stale) { std::cout << (stale?"STREAM_STALE":"STREAM_LIVE") << std::endl; was_stale=stale; }
      if(renderer) {
        int w,h; glfwGetFramebufferSize(glfw.window,&w,&h);
        glViewport(0,0,w,h); glClearColor(stale?0.12f:0,0,0,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        if(!stale) {
          int pw=std::min(w/3,int(h*double(current.color().width())/current.color().height()));
          int ph=int(pw*double(current.color().height())/current.color().width());
          int x=(w-3*pw)/2,y=(h-ph)/2;
          renderer->draw_original(x,y,pw,ph);
          renderer->draw_eye(eye_matrix(current.color_intrinsics(),config,-config.eye_separation_m/2),x+pw,y,pw,ph);
          renderer->draw_eye(eye_matrix(current.color_intrinsics(),config,config.eye_separation_m/2),x+2*pw,y,pw,ph);
          if(!snapshot.empty() && processed>=30 && w>0 && h>0) {
            std::vector<char> pixels(size_t(w)*h*3); glPixelStorei(GL_PACK_ALIGNMENT,1);
            glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());
            std::ofstream file(snapshot,std::ios::binary);
            file << "P6\n" << w << ' ' << h << "\n255\n";
            for(int row=h-1;row>=0;--row) file.write(pixels.data()+size_t(row)*w*3,w*3);
            if(!file) throw std::runtime_error("failed to save snapshot: "+snapshot);
            snapshot.clear();
          }
        }
        glfwSwapBuffers(glfw.window); glfwPollEvents();
        if(glfwGetKey(glfw.window,GLFW_KEY_ESCAPE)==GLFW_PRESS) break;
      }
      if(now-report>=std::chrono::seconds(5)) {
        uint64_t count,replaced;
        {std::lock_guard<std::mutex> lock(mailbox->mutex); count=mailbox->received_count; replaced=mailbox->replaced;}
        const double fps=(count-previous_count)/std::chrono::duration<double>(now-report).count();
        std::ostringstream status;
        status << std::fixed << std::setprecision(2) << (stale?"STALE | ":"") << "RGB | LEFT | RIGHT; rx_fps=" << fps << " valid_depth=" << valid_ratio*100 << "%";
        if(glfw.window) glfwSetWindowTitle(glfw.window,status.str().c_str());
        std::cout << status.str() << " processed=" << processed << " replaced=" << replaced << " frame_gaps=" << frame_gaps << " invalid=" << invalid << " max_skew_ms=" << max_skew << " max_processing_age_ms=" << max_age_ms << std::endl;
        previous_count=count; report=now;
      }
      if(payload.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    messages.close();
    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    std::cout << "SUMMARY seconds=" << elapsed << " received=" << mailbox->received_count << " processed=" << processed << " average_rx_fps=" << mailbox->received_count/elapsed << " replaced=" << mailbox->replaced << " frame_gaps=" << frame_gaps << " invalid=" << invalid << " max_skew_ms=" << max_skew << " max_processing_age_ms=" << max_age_ms << std::endl;
    return processed>0 && invalid==0?0:2;
  } catch(const std::exception& e) { std::cerr << "rgbd_stereo_preview: " << e.what() << '\n'; return 1; }
}
