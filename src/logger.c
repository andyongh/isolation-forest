/*
This is a simple, minimal C logging library for easy reuse.

MIT License

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

#include "logger.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    OutputHandler handler;
    void* ctx;
} HandlerEntry;

struct Logger {
    LogLevel level;
    HandlerEntry* handlers;
    size_t handler_count;
    pthread_mutex_t lock;
};

static Logger* global_logger       = NULL;
static pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;

Logger* logger_create(LogLevel level)
{
    Logger* logger        = malloc(sizeof(Logger));
    logger->level         = level;
    logger->handlers      = NULL;
    logger->handler_count = 0;
    pthread_mutex_init(&logger->lock, NULL);
    return logger;
}

void logger_free(Logger* logger)
{
    if (!logger) return;

    pthread_mutex_lock(&logger->lock);
    free(logger->handlers);
    pthread_mutex_unlock(&logger->lock);
    pthread_mutex_destroy(&logger->lock);
    free(logger);
}

void logger_set_level(Logger* logger, LogLevel level)
{
    if (logger) {
        pthread_mutex_lock(&logger->lock);
        logger->level = level;
        pthread_mutex_unlock(&logger->lock);
    }
}

void logger_add_handler(Logger* logger, OutputHandler handler, void* ctx)
{
    if (!logger || !handler) return;

    pthread_mutex_lock(&logger->lock);

    size_t new_count           = logger->handler_count + 1;
    HandlerEntry* new_handlers = realloc(logger->handlers,
                                         new_count * sizeof(HandlerEntry));

    if (new_handlers) {
        new_handlers[logger->handler_count] = (HandlerEntry){handler, ctx};
        logger->handlers                    = new_handlers;
        logger->handler_count               = new_count;
    }

    pthread_mutex_unlock(&logger->lock);
}

void logger_remove_handlers(Logger* logger)
{
    if (logger) {
        pthread_mutex_lock(&logger->lock);
        free(logger->handlers);
        logger->handlers      = NULL;
        logger->handler_count = 0;
        pthread_mutex_unlock(&logger->lock);
    }
}

static void format_message(char* buf, size_t size,
                           LogLevel level, const char* file,
                           int line, const char* func,
                           const char* format, va_list args)
{
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm);

    const char* level_str = "";
    switch (level) {
    case LOG_DEBUG:
        level_str = "DEBUG";
        break;
    case LOG_INFO:
        level_str = "INFO";
        break;
    case LOG_WARNING:
        level_str = "WARNING";
        break;
    case LOG_ERROR:
        level_str = "ERROR";
        break;
    case LOG_CRITICAL:
        level_str = "CRITICAL";
        break;
    }

    size_t offset = snprintf(buf, size, "[%s][%s] %s:%d (%s) ",
                             timestamp, level_str, file, line, func);
    if (offset > 0 && offset < size) {
        vsnprintf(buf + offset, size - offset, format, args);
    }
}

void logger_log(const Logger* logger, LogLevel level,
                const char* file, int line, const char* func,
                const char* format, ...)
{
    if (!logger || level < logger->level) return;

    char buffer[2048];
    va_list args;
    va_start(args, format);
    format_message(buffer, sizeof(buffer), level, file, line, func, format, args);
    va_end(args);

    pthread_mutex_lock(&((Logger*)logger)->lock);
    for (size_t i = 0; i < logger->handler_count; i++) {
        HandlerEntry entry = logger->handlers[i];
        entry.handler(buffer, entry.ctx);
    }
    pthread_mutex_unlock(&((Logger*)logger)->lock);
}

// default output
void stdio_handler(const char* msg, void* fp)
{
    FILE* stream = fp ? fp : stderr;
    fprintf(stream, "%s\n", msg);
    fflush(stream);
}

void file_handler(const char* msg, void* fp)
{
    if (fp) {
        fprintf((FILE*)fp, "%s\n", msg);
        fflush((FILE*)fp);
    }
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void null_handler(const char* msg, void* ctx)
{
    // silence logging
}

// for global
Logger* logger_global(void)
{
    pthread_mutex_lock(&global_lock);
    if (!global_logger) {
        global_logger = logger_create(LOG_INFO);
        logger_add_handler(global_logger, stdio_handler, stderr);
    }
    pthread_mutex_unlock(&global_lock);
    return global_logger;
}

void logger_global_set(Logger* logger)
{
    pthread_mutex_lock(&global_lock);
    if (global_logger) {
        logger_free(global_logger);
    }
    global_logger = logger;
    pthread_mutex_unlock(&global_lock);
}