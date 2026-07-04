/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Cross-process plumbing for the forked producer/consumer tests: SCM_RIGHTS
 * fd passing over a socketpair, and the fork harness (child runs `child`,
 * parent runs `parent`, exit code combines both).
 */

#pragma once

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static inline bool send_with_fds(int sock, const void* data, size_t len,
                                 const int* fds, int nfd) {
    struct iovec iov = {const_cast<void*>(data), len};
    char cbuf[CMSG_SPACE(8 * sizeof(int))] = {};
    struct msghdr msg = {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    if (nfd > 0) {
        msg.msg_control = cbuf;
        msg.msg_controllen = CMSG_SPACE((size_t)nfd * sizeof(int));
        struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN((size_t)nfd * sizeof(int));
        memcpy(CMSG_DATA(cm), fds, (size_t)nfd * sizeof(int));
    }
    return sendmsg(sock, &msg, 0) == (ssize_t)len;
}

static inline bool recv_with_fds(int sock, void* data, size_t len, int* fds,
                                 int nfd) {
    struct iovec iov = {data, len};
    char cbuf[CMSG_SPACE(8 * sizeof(int))] = {};
    struct msghdr msg = {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);
    if (recvmsg(sock, &msg, 0) != (ssize_t)len)
        return false;
    for (int i = 0; i < nfd; i++)
        fds[i] = -1;
    for (struct cmsghdr* cm = CMSG_FIRSTHDR(&msg); cm;
         cm = CMSG_NXTHDR(&msg, cm)) {
        if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) {
            int got = (int)((cm->cmsg_len - CMSG_LEN(0)) / sizeof(int));
            for (int i = 0; i < nfd && i < got; i++)
                memcpy(&fds[i], CMSG_DATA(cm) + i * sizeof(int), sizeof(int));
            return true;
        }
    }
    return nfd == 0;
}

static inline bool send_with_fd(int sock, const void* data, size_t len, int fd) {
    return send_with_fds(sock, data, len, &fd, 1);
}

static inline bool recv_with_fd(int sock, void* data, size_t len, int* fd) {
    return recv_with_fds(sock, data, len, fd, 1);
}

/* Fork harness: the child runs `child(sock)`, the parent runs `parent(sock)`,
 * then the child is reaped. Each side does its own dmn_init. Returns 0 only
 * when both sides returned 0. */
static inline int run_fork_pair(const char* tag, int (*parent)(int),
                                int (*child)(int)) {
    /* Unbuffered so parent and child output stays ordered and survives the
     * child's _exit (which skips stdio flushing). */
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "%s: socketpair FAILED: %s\n", tag, strerror(errno));
        return 1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "%s: fork FAILED: %s\n", tag, strerror(errno));
        return 1;
    }
    if (pid == 0) {
        close(sv[0]);
        int rc = child(sv[1]);
        close(sv[1]);
        _exit(rc);
    }
    close(sv[1]);
    int prc = parent(sv[0]);
    close(sv[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    int crc = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    if (prc != 0 || crc != 0) {
        fprintf(stderr, "%s: FAIL (parent=%d child=%d)\n", tag, prc, crc);
        return 1;
    }
    return 0;
}
