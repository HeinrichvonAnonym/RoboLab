#pragma once

namespace robo_lab {

/// Base type for RoboLab components (lifecycle hooks shared across the repo).
class RoboLab {
 public:
  virtual ~RoboLab();

  virtual void initialize();
  virtual void run();
  virtual void close();
};

}  // namespace robo_lab
