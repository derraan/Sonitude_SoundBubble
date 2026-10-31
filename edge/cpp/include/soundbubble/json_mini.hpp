#pragma once

#include <cctype>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace soundbubble {

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Json> a;
  std::map<std::string, Json> o;

  const Json* find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    auto it = o.find(key);
    return it == o.end() ? nullptr : &it->second;
  }
  bool has(const std::string& key) const { return find(key) != nullptr; }
};

inline void skip_ws(std::string_view t, size_t& i) {
  while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
}

inline Json parse_json_at(std::string_view t, size_t& i);

inline std::string parse_string(std::string_view t, size_t& i) {
  if (i >= t.size() || t[i] != '"') throw std::runtime_error("json string");
  ++i;
  std::string out;
  while (i < t.size()) {
    char c = t[i++];
    if (c == '"') return out;
    if (c == '\\' && i < t.size()) {
      char e = t[i++];
      if (e == 'n') out.push_back('\n');
      else if (e == 't') out.push_back('\t');
      else if (e == 'r') out.push_back('\r');
      else out.push_back(e);
    } else {
      out.push_back(c);
    }
  }
  throw std::runtime_error("unterminated json string");
}

inline Json parse_json_at(std::string_view t, size_t& i) {
  skip_ws(t, i);
  if (i >= t.size()) throw std::runtime_error("unexpected end of json");
  Json j;
  char c = t[i];
  if (c == '"') {
    j.type = Json::Type::String;
    j.s = parse_string(t, i);
    return j;
  }
  if (c == '{') {
    j.type = Json::Type::Object;
    ++i;
    skip_ws(t, i);
    if (i < t.size() && t[i] == '}') {
      ++i;
      return j;
    }
    while (i < t.size()) {
      skip_ws(t, i);
      std::string key = parse_string(t, i);
      skip_ws(t, i);
      if (i >= t.size() || t[i] != ':') throw std::runtime_error("json object colon");
      ++i;
      j.o.emplace(key, parse_json_at(t, i));
      skip_ws(t, i);
      if (i < t.size() && t[i] == ',') {
        ++i;
        continue;
      }
      if (i < t.size() && t[i] == '}') {
        ++i;
        return j;
      }
      throw std::runtime_error("json object");
    }
  }
  if (c == '[') {
    j.type = Json::Type::Array;
    ++i;
    skip_ws(t, i);
    if (i < t.size() && t[i] == ']') {
      ++i;
      return j;
    }
    while (i < t.size()) {
      j.a.push_back(parse_json_at(t, i));
      skip_ws(t, i);
      if (i < t.size() && t[i] == ',') {
        ++i;
        continue;
      }
      if (i < t.size() && t[i] == ']') {
        ++i;
        return j;
      }
      throw std::runtime_error("json array");
    }
  }
  if (t.compare(i, 4, "true") == 0) {
    j.type = Json::Type::Bool;
    j.b = true;
    i += 4;
    return j;
  }
  if (t.compare(i, 5, "false") == 0) {
    j.type = Json::Type::Bool;
    j.b = false;
    i += 5;
    return j;
  }
  if (t.compare(i, 4, "null") == 0) {
    i += 4;
    return j;
  }
  size_t start = i;
  if (c == '-' || c == '+') ++i;
  while (i < t.size() && (std::isdigit(static_cast<unsigned char>(t[i])) || t[i] == '.' || t[i] == 'e' || t[i] == 'E' ||
                          t[i] == '+' || t[i] == '-')) {
    ++i;
  }
  if (start == i) throw std::runtime_error("json value");
  j.type = Json::Type::Number;
  j.n = std::stod(std::string(t.substr(start, i - start)));
  return j;
}

inline Json parse_json(std::string_view t) {
  size_t i = 0;
  Json j = parse_json_at(t, i);
  return j;
}

inline void dump_json_impl(std::ostream& o, const Json& j) {
  switch (j.type) {
    case Json::Type::Null:
      o << "null";
      break;
    case Json::Type::Bool:
      o << (j.b ? "true" : "false");
      break;
    case Json::Type::Number:
      o << j.n;
      break;
    case Json::Type::String:
      o << '"';
      for (char c : j.s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c == '\n') o << "\\n";
        else o << c;
      }
      o << '"';
      break;
    case Json::Type::Array:
      o << '[';
      for (size_t i = 0; i < j.a.size(); ++i) {
        if (i) o << ',';
        dump_json_impl(o, j.a[i]);
      }
      o << ']';
      break;
    case Json::Type::Object:
      o << '{';
      {
        bool first = true;
        for (const auto& kv : j.o) {
          if (!first) o << ',';
          first = false;
          Json key;
          key.type = Json::Type::String;
          key.s = kv.first;
          dump_json_impl(o, key);
          o << ':';
          dump_json_impl(o, kv.second);
        }
      }
      o << '}';
      break;
  }
}

inline std::string dump_json(const Json& j) {
  std::ostringstream o;
  dump_json_impl(o, j);
  return o.str();
}

inline Json json_string(std::string s) {
  Json j;
  j.type = Json::Type::String;
  j.s = std::move(s);
  return j;
}
inline Json json_number(double n) {
  Json j;
  j.type = Json::Type::Number;
  j.n = n;
  return j;
}
inline Json json_bool(bool b) {
  Json j;
  j.type = Json::Type::Bool;
  j.b = b;
  return j;
}

}  // namespace soundbubble
