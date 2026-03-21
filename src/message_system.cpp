#include "message_system.h"

#include <unordered_map>
#include <vector>

#include "zenoh.h"

namespace robo_lab {

namespace {

struct CallbackCtx {
  std::function<void(const std::string&, const std::string&)> cb;
};

void callback_ctx_drop(void* arg) {
  delete static_cast<CallbackCtx*>(arg);
}

void sample_handler(z_loaned_sample_t* sample, void* arg) {
  auto* ctx = static_cast<CallbackCtx*>(arg);
  if (ctx == nullptr || sample == nullptr) {
    return;
  }

  z_view_string_t key_view;
  z_keyexpr_as_view_string(z_sample_keyexpr(sample), &key_view);
  const z_loaned_string_t* key_str = z_loan(key_view);

  z_owned_string_t payload_owned;
  z_bytes_to_string(z_sample_payload(sample), &payload_owned);
  const z_loaned_string_t* payload_str = z_loan(payload_owned);

  std::string key(z_string_data(key_str), z_string_len(key_str));
  std::string payload(z_string_data(payload_str), z_string_len(payload_str));

  ctx->cb(key, payload);

  z_drop(z_move(payload_owned));
}

struct PublisherSlot {
  z_owned_publisher_t pub{};
  ~PublisherSlot() { z_drop(z_move(pub)); }
};

struct SubscriberSlot {
  z_owned_subscriber_t sub{};
  ~SubscriberSlot() { z_drop(z_move(sub)); }
};

}  // namespace

struct MessageSystem::Impl {
  z_owned_session_t session{};
  bool session_open = false;
  std::unordered_map<std::string, std::unique_ptr<PublisherSlot>> publishers;
  std::vector<std::unique_ptr<SubscriberSlot>> subscribers;
};

MessageSystem::MessageSystem() : impl_(std::make_unique<Impl>()) {}

MessageSystem::~MessageSystem() { close(); }

void MessageSystem::initialize() {
  if (impl_->session_open) {
    return;
  }

  zc_init_log_from_env_or("error");

  z_owned_config_t config;
  z_config_default(&config);
  if (z_open(&impl_->session, z_move(config), nullptr) < 0) {
    return;
  }

  impl_->session_open = true;
}

void MessageSystem::close() {
  impl_->publishers.clear();
  impl_->subscribers.clear();

  if (impl_->session_open) {
    z_drop(z_move(impl_->session));
    impl_->session_open = false;
  }
}

bool MessageSystem::is_open() const { return impl_->session_open; }

bool MessageSystem::publish(const std::string& keyexpr, const std::string& payload) {
  if (!impl_->session_open) {
    return false;
  }

  PublisherSlot* slot = nullptr;
  auto it = impl_->publishers.find(keyexpr);
  if (it == impl_->publishers.end()) {
    auto owned = std::make_unique<PublisherSlot>();
    z_view_keyexpr_t ke{};
    if (z_view_keyexpr_from_str(&ke, keyexpr.c_str()) != Z_OK) {
      return false;
    }
    if (z_declare_publisher(z_loan(impl_->session), &owned->pub, z_loan(ke), nullptr) < 0) {
      return false;
    }
    slot = owned.get();
    impl_->publishers.emplace(keyexpr, std::move(owned));
  } else {
    slot = it->second.get();
  }

  z_publisher_put_options_t options{};
  z_publisher_put_options_default(&options);

  z_owned_bytes_t bytes{};
  if (z_bytes_copy_from_buf(&bytes, reinterpret_cast<const uint8_t*>(payload.data()), payload.size()) < 0) {
    return false;
  }

  z_owned_encoding_t encoding{};
  z_encoding_clone(&encoding, z_encoding_text_plain());
  options.encoding = z_move(encoding);

  const int rc = z_publisher_put(z_loan(slot->pub), z_move(bytes), &options);
  return rc == Z_OK;
}

bool MessageSystem::subscribe(
    const std::string& keyexpr,
    std::function<void(const std::string& key, const std::string& payload)> callback) {
  if (!impl_->session_open) {
    return false;
  }

  z_view_keyexpr_t ke{};
  if (z_view_keyexpr_from_str(&ke, keyexpr.c_str()) != Z_OK) {
    return false;
  }

  auto* ctx = new CallbackCtx{std::move(callback)};
  z_owned_closure_sample_t closure{};
  z_closure(&closure, sample_handler, callback_ctx_drop, ctx);

  auto sub = std::make_unique<SubscriberSlot>();
  if (z_declare_subscriber(z_loan(impl_->session), &sub->sub, z_loan(ke), z_move(closure), nullptr) < 0) {
    delete ctx;
    return false;
  }

  impl_->subscribers.push_back(std::move(sub));
  return true;
}

}  // namespace robo_lab
