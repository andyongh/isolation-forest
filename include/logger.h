/*
This is a simple, minimal C logging library for easy reuse.

The MIT License (MIT)

Copyright (c) 2025 AndY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#ifndef LOGGER_H
#define LOGGER_H

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_CRITICAL
} LogLevel;

typedef struct Logger Logger;

typedef void (*OutputHandler)(const char* message, void* ctx);

Logger* logger_create(LogLevel level);
void logger_free(Logger* logger);

void logger_set_level(Logger* logger, LogLevel level);
void logger_add_handler(Logger* logger, OutputHandler handler, void* ctx);
void logger_remove_handlers(Logger* logger);

void logger_log(const Logger* logger, LogLevel level,
                const char* file, int line, const char* func,
                const char* format, ...);

void stdio_handler(const char* msg, void* fp);
void file_handler(const char* msg, void* fp);
void null_handler(const char* msg, void* ctx);

Logger* logger_global(void);
void logger_global_set(Logger* logger);

#define LOG_DEBUG(...)    logger_log(logger_global(), LOG_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...)     logger_log(logger_global(), LOG_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARNING(...)  logger_log(logger_global(), LOG_WARNING, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...)    logger_log(logger_global(), LOG_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_CRITICAL(...) logger_log(logger_global(), LOG_CRITICAL, __FILE__, __LINE__, __func__, __VA_ARGS__)

#endif  // LOGGER_H