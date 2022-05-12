iblocksync
==========

`iblocksync` incrementally syncs raw disk images. It uses separate server processes for serving (i.e. reading) the source image and receiving (i.e. writing) the target image, allowing one to transparently read from and write to remote systems through SSH connections. Naturally it also supports syncing images on the local machine. `iblocksync` furthermore allows wrapping its remote processes, allowing one to do custom tasks before and after syncing, e.g. creating and removing LVM snapshots.

Install
-------

You can install `iblocksync` just regularily using `distutils` and the provided `setup.py`. It doesn't depend on any external modules, just Python core modules.

Please note that you must install `iblocksync` on the local system as well as all remote systems.

`iblocksync` works with Python 3 only. It was tested with Python 3.4 under Debian Jessie, however, it *should* work with any other distribution. If not, please don't hesitate to open a new [Issue on GitHub](https://github.com/PhrozenByte/iblocksync/issues).

Usage
-----

```
$ iblocksync --help
usage: iblocksync [OPTION]... SOURCE TARGET

Incrementally syncs a raw disk image from SOURCE to TARGET.

Arguments:
  SOURCE                Path to the raw disk image to sync (i.e. the source
                        image). You can either specify a absolute path
                        ('/path/to/image'), a relative path ('path/to/image'),
                        a scp remote path ('user@host:/path/to/image', 'user@'
                        being optional), or URL-style SSH paths
                        ('ssh://user@host:port/path/to/image', both 'user@'
                        and ':port' being optional). URL-style paths are
                        absolute by default, however, you can use
                        'ssh://user@host:port/./path/to/image' to make them
                        relative. You can also specify paths relative to
                        user's ('~/path/to/image') or other's home directory
                        ('~other/path/to/image')
  TARGET                Path to the raw disk image to sync to (i.e. the target
                        image). You can specify the same path syntax as for
                        the source image (see above)

Application options:
  -a HASH_ALGO, --algo HASH_ALGO
                        Gives the hash algorithm to work with. The following
                        hash algorithms are supported: 'adler32', 'sha1',
                        'sha224', 'sha256', 'sha512', 'sha384', 'md5',
                        'crc32'. Defaults to 'sha1'
  -b BLOCK_SIZE, --block BLOCK_SIZE
                        Gives the block size to work with. A suffix of B for
                        bytes, K for kilobytes (1,024 bytes), M for megabytes
                        (1,024 * 1,024 bytes), G for gigabytes, T for
                        terabytes, P for petabytes or E for exabytes is
                        optional. Use kB/MB/GB/TB/PB/EB for base-10 units (1
                        kB = 1,000 bytes, 1 MB = 1,000 * 1,000 bytes, ...).
                        Defaults to 1M (1,048,576 bytes)
  -p PROGRESS, --progress PROGRESS
                        Shows a simple progress bar. The progress bar is
                        disabled when verbosity was increased twice. The
                        following progress bar styles are supported: 'none',
                        'dot', 'full'. Defaults to 'full'
  -f, --force           Ignore warnings, never prompt
  -v, --verbose         Explain what is being done (increase verbosity)

Help options:
  --help                Display this help and exit
  --version             Output version information and exit

You can manipulate the way how iblocksync communicates with its remote
processes using env variables. Pass the 'IBLOCKSYNC_SOURCE_EXEC' env variable
to overwrite the command iblocksync executes on the source system (defaults to
'iblocksync-serve'). If your source image is on a remote system, you can also
overwrite the command iblocksync uses to establish the connection using the
'IBLOCKSYNC_SOURCE_RSH' env variable (defaults to 'ssh'). This for example
allows you to pass an identity file to ssh ('ssh -i ~/.ssh/id_dsa'). The same
options apply to the target system: Set 'IBLOCKSYNC_TARGET_EXEC' (defaults to
'iblocksync-receive') and/or 'IBLOCKSYNC_TARGET_RSH' (defaults to 'ssh')
respectively.
```

License & Copyright
-------------------

Copyright (C) 2017-2022  Daniel Rudolf <http://www.daniel-rudolf.de/>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU General Public License](LICENSE) for more details.
