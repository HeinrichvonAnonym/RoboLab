// Real PeerConnections, SCTP/DTLS and H.264 encode/decode; no camera or signaling service.
#include "api/create_peerconnection_factory.h"
#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/jsep.h"
#include "api/make_ref_counted.h"
#include "api/peer_connection_interface.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_broadcaster.h"
#include "api/video/video_frame.h"
#include "api/video_codecs/video_encoder_factory_template.h"
#include "api/video_codecs/video_encoder_factory_template_open_h264_adapter.h"
#include "api/video_codecs/video_decoder_factory_template.h"
#include "api/video_codecs/video_decoder_factory_template_open_h264_adapter.h"
#include "pc/video_track_source.h"
#include "modules/audio_device/include/audio_device_default.h"
#include "rtc_base/logging.h"
#include "rtc_base/ssl_adapter.h"
#include "rtc_base/thread.h"
#include "rtc_base/time_utils.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace webrtc;
using namespace std::chrono_literals;
void check(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class T> T await(std::future<T>& future,const char* operation) {
  check(future.wait_for(10s)==std::future_status::ready,std::string(operation)+" timed out");
  return future.get();
}
class CreateDescription : public CreateSessionDescriptionObserver {
 public:
  std::promise<std::unique_ptr<SessionDescriptionInterface>> result;
  void OnSuccess(SessionDescriptionInterface* description) override { result.set_value(std::unique_ptr<SessionDescriptionInterface>(description)); }
  void OnFailure(RTCError error) override { result.set_exception(std::make_exception_ptr(std::runtime_error(error.message()))); }
};
class SetDescription : public SetSessionDescriptionObserver {
 public:
  std::promise<bool> result;
  void OnSuccess() override { result.set_value(true); }
  void OnFailure(RTCError error) override { result.set_exception(std::make_exception_ptr(std::runtime_error(error.message()))); }
};
class Source : public VideoTrackSource {
 public:
  Source():VideoTrackSource(false) {}
  void push(int sequence) {
    auto buffer=I420Buffer::Create(640,480);
    for(int y=0;y<480;++y) {
      auto* row=buffer->MutableDataY()+y*buffer->StrideY();
      std::memset(row,70,320);std::memset(row+320,180,320);
      // A moving stripe ensures the decoder receives changing frames, not a cached image.
      row[(sequence*3)%640]=230;
    }
    for(int y=0;y<240;++y) {
      std::memset(buffer->MutableDataU()+y*buffer->StrideU(),128,320);
      std::memset(buffer->MutableDataV()+y*buffer->StrideV(),128,320);
    }
    broadcaster_.OnFrame(VideoFrame::Builder().set_video_frame_buffer(buffer).set_timestamp_us(TimeMicros()).set_rotation(kVideoRotation_0).build());
  }
 private:
  VideoSourceInterface<VideoFrame>* source() override { return &broadcaster_; }
  VideoBroadcaster broadcaster_;
};
struct Sink : VideoSinkInterface<VideoFrame> {
  std::atomic<int> decoded{0},valid{0};
  void OnFrame(const VideoFrame& frame) override {
    ++decoded;
    auto buffer=frame.video_frame_buffer()->ToI420();
    if(buffer->width()==640 && buffer->height()==480) {
      int left=buffer->DataY()[240*buffer->StrideY()+160],right=buffer->DataY()[240*buffer->StrideY()+480];
      if(std::abs(left-70)<20 && std::abs(right-180)<20) ++valid;
    }
  }
};
struct ChannelObserver : DataChannelObserver {
  std::mutex mutex;
  std::string received;
  bool binary=false;
  void OnStateChange() override {}
  void OnMessage(const DataBuffer& buffer) override {
    std::lock_guard<std::mutex> lock(mutex);
    received.assign(reinterpret_cast<const char*>(buffer.data.data()),buffer.data.size());binary=buffer.binary;
  }
  bool matches(const std::string& data) {std::lock_guard<std::mutex> lock(mutex);return binary && received==data;}
};
struct PeerObserver : PeerConnectionObserver {
  Sink sink;
  ChannelObserver data;
  scoped_refptr<DataChannelInterface> channel;
  scoped_refptr<VideoTrackInterface> incoming;
  std::atomic<bool> gathered{false};
  void OnSignalingChange(PeerConnectionInterface::SignalingState) override {}
  void OnDataChannel(scoped_refptr<DataChannelInterface> dc) override { channel=dc;channel->RegisterObserver(&data); }
  void OnIceGatheringChange(PeerConnectionInterface::IceGatheringState state) override {gathered=state==PeerConnectionInterface::kIceGatheringComplete;}
  void OnIceCandidate(const IceCandidate*) override {} // Full local SDP is exchanged after gathering.
  void OnTrack(scoped_refptr<RtpTransceiverInterface> transceiver) override {
    auto track=transceiver->receiver()->track();
    if(track->kind()==MediaStreamTrackInterface::kVideoKind) {
      incoming=scoped_refptr<VideoTrackInterface>(static_cast<VideoTrackInterface*>(track.get()));
      incoming->AddOrUpdateSink(&sink,VideoSinkWants{});
    }
  }
  void detach() {if(channel)channel->UnregisterObserver();if(incoming)incoming->RemoveSink(&sink);channel=nullptr;incoming=nullptr;}
};
std::unique_ptr<SessionDescriptionInterface> create_sdp(PeerConnectionInterface* peer,bool offer) {
  auto observer=make_ref_counted<CreateDescription>();auto future=observer->result.get_future();
  if(offer)peer->CreateOffer(observer.get(),{});else peer->CreateAnswer(observer.get(),{});
  return await(future,"create SDP");
}
void set_sdp(PeerConnectionInterface* peer,std::unique_ptr<SessionDescriptionInterface> sdp,bool local) {
  auto observer=make_ref_counted<SetDescription>();auto future=observer->result.get_future();
  if(local)peer->SetLocalDescription(observer.get(),sdp.release());else peer->SetRemoteDescription(observer.get(),sdp.release());
  await(future,"set SDP");
}
void wait_gather(PeerObserver& observer) {
  auto deadline=std::chrono::steady_clock::now()+10s;
  while(!observer.gathered && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(10ms);
  check(observer.gathered,"ICE gathering timed out");
}
std::unique_ptr<SessionDescriptionInterface> copy_local(Thread* signaling,PeerConnectionInterface* peer,SdpType type) {
  std::string text;
  signaling->BlockingCall([&]{peer->local_description()->ToString(&text);});
  check(text.find("H264/90000")!=std::string::npos,"H.264 missing from negotiated SDP");
  SdpParseError error;auto result=CreateSessionDescription(type,text,&error);
  check(bool(result),"SDP parse: "+error.description);return result;
}
void round_trip(int round,Thread* network,Thread* signaling) {
  PeerObserver alice_observer,bob_observer;
  // The standard media factory requires audio dependencies even for a video-only
  // call. A no-op ADM keeps this test independent of microphones/audio services.
  auto adm=make_ref_counted<webrtc_impl::AudioDeviceModuleDefault<AudioDeviceModule>>();
  auto factory=CreatePeerConnectionFactory(network,signaling,adm,
    CreateBuiltinAudioEncoderFactory(),CreateBuiltinAudioDecoderFactory(),
    std::make_unique<VideoEncoderFactoryTemplate<OpenH264EncoderTemplateAdapter>>(),
    std::make_unique<VideoDecoderFactoryTemplate<OpenH264DecoderTemplateAdapter>>(),nullptr,nullptr);
  check(bool(factory),"PeerConnectionFactory creation failed");
  PeerConnectionInterface::RTCConfiguration config;
  config.sdp_semantics=SdpSemantics::kUnifiedPlan;
  auto ar=factory->CreatePeerConnectionOrError(config,PeerConnectionDependencies(&alice_observer));
  check(ar.ok(),ar.ok()?"":ar.error().message());auto alice=ar.MoveValue();
  auto br=factory->CreatePeerConnectionOrError(config,PeerConnectionDependencies(&bob_observer));
  check(br.ok(),br.ok()?"":br.error().message());auto bob=br.MoveValue();
  scoped_refptr<Source> source;
  scoped_refptr<VideoTrackInterface> track;
  auto cleanup=[&]{
    signaling->BlockingCall([&]{alice_observer.detach();bob_observer.detach();});
    alice->Close();bob->Close();alice=nullptr;bob=nullptr;track=nullptr;source=nullptr;factory=nullptr;
  };
  try {
    signaling->BlockingCall([&]{source=make_ref_counted<Source>();});
    track=factory->CreateVideoTrack(source,"synthetic-h264");
    auto sender=alice->AddTrack(track,{"smoke"});check(sender.ok(),sender.ok()?"":sender.error().message());
    DataChannelInit dc_config;
    auto dc=alice->CreateDataChannelOrError("binary-protobuf-smoke",&dc_config);check(dc.ok(),dc.ok()?"":dc.error().message());
    signaling->BlockingCall([&]{alice_observer.channel=dc.MoveValue();alice_observer.channel->RegisterObserver(&alice_observer.data);});
    set_sdp(alice.get(),create_sdp(alice.get(),true),true);wait_gather(alice_observer);
    set_sdp(bob.get(),copy_local(signaling,alice.get(),SdpType::kOffer),false);
    set_sdp(bob.get(),create_sdp(bob.get(),false),true);wait_gather(bob_observer);
    set_sdp(alice.get(),copy_local(signaling,bob.get(),SdpType::kAnswer),false);
    std::string payload(4096,'\0');for(size_t i=0;i<payload.size();++i)payload[i]=static_cast<char>((i*71+round)%256);
    bool sent=false,echoed=false;int sequence=0;
    auto deadline=std::chrono::steady_clock::now()+12s;
    while(std::chrono::steady_clock::now()<deadline) {
      signaling->BlockingCall([&]{
        if(!sent && alice_observer.channel->state()==DataChannelInterface::kOpen)sent=alice_observer.channel->Send(DataBuffer(CopyOnWriteBuffer(payload.data(),payload.size()),true));
        if(sent && !echoed && bob_observer.data.matches(payload) && bob_observer.channel && bob_observer.channel->state()==DataChannelInterface::kOpen)
          echoed=bob_observer.channel->Send(DataBuffer(CopyOnWriteBuffer(payload.data(),payload.size()),true));
      });
      source->push(sequence++);
      if(echoed && alice_observer.data.matches(payload) && bob_observer.sink.valid>=30)break;
      std::this_thread::sleep_for(33ms);
    }
    check(echoed && alice_observer.data.matches(payload),"binary DataChannel round-trip failed");
    check(bob_observer.sink.valid>=30,"H.264 sink did not decode 30 correct 640x480 frames");
    std::cout << "PASS round=" << round << " binary_bytes=" << payload.size() << " codec=H264 decoded=" << bob_observer.sink.decoded << " validated=" << bob_observer.sink.valid << std::endl;
    cleanup();
  } catch(...) {cleanup();throw;}
}
}
int main() {
  webrtc::LogMessage::LogToDebug(webrtc::LS_WARNING);
  if(!webrtc::InitializeSSL())return 1;
  auto network=webrtc::Thread::CreateWithSocketServer(),signaling=webrtc::Thread::Create();
  network->Start();signaling->Start();int result=0;
  try {for(int i=1;i<=3;++i)round_trip(i,network.get(),signaling.get());}
  catch(const std::exception& e){std::cerr << "FAIL " << e.what() << std::endl;result=1;}
  signaling->Stop();network->Stop();webrtc::CleanupSSL();return result;
}
