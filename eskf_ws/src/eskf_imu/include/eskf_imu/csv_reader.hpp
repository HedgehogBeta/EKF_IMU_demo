#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace eskf_imu {

// 按列名读取的 CSV 表。数据里混有一行 NUL 开头的坏行，解析必须严格：
// 字段非空、无 NUL、被完全消费才认。
struct CsvTable {
  std::vector<std::string> header;
  std::unordered_map<std::string, size_t> col_index;
  std::vector<std::vector<double>> rows;  // 每行按表头列序存放
  size_t skipped_rows = 0;
  size_t line_no_of_last_skip = 0;  // 1-based，含表头行

  size_t num_cols() const { return header.size(); }

  const std::vector<double>& row(size_t i) const { return rows[i]; }

  double at(size_t row_i, const std::string& col) const {
    return rows[row_i][col_index.at(col)];
  }
};

inline bool parse_double(const std::string& field, double& out) {
  if (field.empty()) return false;
  for (char c : field) {
    if (c == '\0') return false;
  }
  try {
    size_t pos = 0;
    out = std::stod(field, &pos);
    while (pos < field.size() &&
           (field[pos] == ' ' || field[pos] == '\t' || field[pos] == '\r')) {
      ++pos;
    }
    return pos == field.size();
  } catch (const std::exception&) {
    return false;
  }
}

inline std::vector<std::string> split_line(const std::string& line) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    size_t comma = line.find(',', start);
    if (comma == std::string::npos) {
      fields.push_back(line.substr(start));
      break;
    }
    fields.push_back(line.substr(start, comma - start));
    start = comma + 1;
  }
  for (auto& f : fields) {
    while (!f.empty() && (f.back() == '\r' || f.back() == '\n')) f.pop_back();
  }
  return fields;
}

// 任何一行有一列解析失败就整行跳过并计数
inline CsvTable read_csv(const std::string& path) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("无法打开 CSV: " + path);
  }
  CsvTable table;
  std::string line;
  if (!std::getline(fin, line)) {
    throw std::runtime_error("CSV 为空: " + path);
  }
  table.header = split_line(line);
  for (size_t i = 0; i < table.header.size(); ++i) {
    table.col_index[table.header[i]] = i;
  }

  size_t line_no = 1;
  while (std::getline(fin, line)) {
    ++line_no;
    if (line.empty()) continue;
    std::vector<std::string> fields = split_line(line);
    if (fields.size() != table.header.size()) {
      ++table.skipped_rows;
      table.line_no_of_last_skip = line_no;
      continue;
    }
    std::vector<double> row(fields.size());
    bool ok = true;
    for (size_t i = 0; i < fields.size(); ++i) {
      if (!parse_double(fields[i], row[i])) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      ++table.skipped_rows;
      table.line_no_of_last_skip = line_no;
      continue;
    }
    table.rows.push_back(std::move(row));
  }
  return table;
}

}  // namespace eskf_imu
