#ifndef COLORS_HPP
#define COLORS_HPP

#pragma once

#include <string>

namespace color {

    inline constexpr const char* RESET = "\033[0m";
    inline constexpr const char* BLACK = "\033[30m";
    inline constexpr const char* RED = "\033[31m";
    inline constexpr const char* GREEN = "\033[32m";
    inline constexpr const char* YELLOW = "\033[33m";
    inline constexpr const char* BLUE = "\033[34m";
    inline constexpr const char* MAGENTA = "\033[35m";
    inline constexpr const char* CYAN = "\033[36m";
    inline constexpr const char* WHITE = "\033[37m";

    inline constexpr const char* BOLD_BLACK = "\033[1m\033[30m";
    inline constexpr const char* BOLD_RED = "\033[1m\033[31m";
    inline constexpr const char* BOLD_GREEN = "\033[1m\033[32m";
    inline constexpr const char* BOLD_YELLOW = "\033[1m\033[33m";
    inline constexpr const char* BOLD_BLUE = "\033[1m\033[34m";
    inline constexpr const char* BOLD_MAGENTA = "\033[1m\033[35m";
    inline constexpr const char* BOLD_CYAN = "\033[1m\033[36m";
    inline constexpr const char* BOLD_WHITE = "\033[1m\033[37m";

    inline std::string wrap(const std::string& text, const char* code) {
        return std::string(code) + text + RESET;
    }

    inline std::string red(const std::string& text) { return wrap(text, RED); }
    inline std::string green(const std::string& text) { return wrap(text, GREEN); }
    inline std::string blue(const std::string& text) { return wrap(text, BLUE); }
    inline std::string yellow(const std::string& text) { return wrap(text, YELLOW); }
    inline std::string cyan(const std::string& text) { return wrap(text, CYAN); }
    inline std::string magenta(const std::string& text) { return wrap(text, MAGENTA); }
    inline std::string bold_red(const std::string& text) { return wrap(text, BOLD_RED); }
    inline std::string bold_green(const std::string& text) { return wrap(text, BOLD_GREEN); }
    inline std::string bold_blue(const std::string& text) { return wrap(text, BOLD_BLUE); }
    inline std::string bold_yellow(const std::string& text) { return wrap(text, BOLD_YELLOW); }
    inline std::string bold_cyan(const std::string& text) { return wrap(text, BOLD_CYAN); }
    inline std::string bold_magenta(const std::string& text) { return wrap(text, BOLD_MAGENTA); }
    inline std::string bold_white(const std::string& text) { return wrap(text, BOLD_WHITE); }

	inline std::string bold(const std::string& text) { return wrap(text, "\033[1m"); }

    inline std::string colorize(const std::string& text, const std::string& colorName) {
        if (colorName == "red") return red(text);
        if (colorName == "green") return green(text);
        if (colorName == "blue") return blue(text);
        if (colorName == "yellow") return yellow(text);
        if (colorName == "cyan") return cyan(text);
        if (colorName == "magenta") return magenta(text);
        if (colorName == "bold_red") return bold_red(text);
        if (colorName == "bold_green") return bold_green(text);
        if (colorName == "bold_blue") return bold_blue(text);
        if (colorName == "bold_yellow") return bold_yellow(text);
        if (colorName == "bold_cyan") return bold_cyan(text);
        if (colorName == "bold_magenta") return bold_magenta(text);
        if (colorName == "bold_white") return bold_white(text);
        if (colorName == "bold") return bold(text);
        return text; // No color applied
	}

    inline std::string debug() { return red("## DEBUG:\n"); }

} // namespace color

#endif // COLORS_HPP
