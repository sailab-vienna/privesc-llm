# Procedural base image for generated privilege escalation scenarios.
# Packages are auto-generated from GTFOBins allowlists.
FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Copy generated package list
COPY packages.txt /tmp/packages.txt

# Install base system dependencies + allowlisted packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      cron \
      binutils \
      bsdextrautils \
      file \
      curl \
      iproute2 \
      lsb-release \
      net-tools \
      openssh-server \
      python-is-python3 \
      sudo \
      $(cat /tmp/packages.txt) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/packages.txt

# Debian login.defs sets ENV_PATH without /usr/sbin for non-root users.
# sshd reads /etc/environment via pam_env, so set a full PATH there.
RUN printf 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n' > /etc/environment

# Prepare SSH daemon
RUN mkdir /var/run/sshd && \
    mkdir -p /etc/ssh/sshd_config.d && \
    printf '%s\n' \
      'MaxSessions 1024' \
      'MaxStartups 2000:30:4000' \
      'ClientAliveInterval 30' \
      'ClientAliveCountMax 3' \
      'LoginGraceTime 1m' \
      'UseDNS no' \
      'X11Forwarding no' \
      > /etc/ssh/sshd_config.d/99-privesc-performance.conf && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

EXPOSE 22
CMD ["/usr/sbin/sshd","-D","-e"]
