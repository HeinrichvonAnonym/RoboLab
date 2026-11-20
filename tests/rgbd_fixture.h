#pragma once
#include "camera/rgbd.h"

inline camera::RgbdFrame rgbd_fixture(int w=8,int h=8,int padding=0) {
  camera::RgbdFrame f;
  f.set_schema_version(1); f.set_serial("synthetic"); f.set_depth_units_m(0.0001f);
  for(auto* image:{f.mutable_color(),f.mutable_depth()}) {
    bool rgb=image==f.mutable_color(); image->set_encoding(rgb?camera::Image::RGB8:camera::Image::Z16);
    image->set_width(w); image->set_height(h); image->set_stride(w*(rgb?3:2)+padding);
    image->set_data(std::string(image->stride()*h,0)); image->set_frame_number(1); image->set_device_timestamp_ms(100);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
      auto* bytes=image->mutable_data()->data()+y*image->stride()+x*(rgb?3:2);
      if(rgb) {bytes[0]=x<w/2?255:0;bytes[1]=0;bytes[2]=x<w/2?0:255;}
      else {bytes[0]=2000&255; bytes[1]=2000>>8;}
    }
  }
  for(auto* k:{f.mutable_color_intrinsics(),f.mutable_depth_intrinsics()}) {
    k->set_width(w); k->set_height(h); k->set_fx(w); k->set_fy(h); k->set_ppx((w-1)*0.5f); k->set_ppy((h-1)*0.5f);
    for(int i=0;i<5;++i)k->add_coefficients(0);
  }
  for(int i=0;i<9;++i) f.mutable_depth_to_color()->add_rotation(i%4==0?1:0);
  for(int i=0;i<3;++i) f.mutable_depth_to_color()->add_translation_m(0);
  return f;
}
