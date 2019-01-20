import argparse, collections, errno, hashlib, itertools, json, logging, os, re, shlex, struct, subprocess, sys, threading, time, zlib

__version__ = "1.0.0"

_SIGNAL_ACK = b'\x06'
_SIGNAL_NAK = b'\x15'
_SIGNAL_EOT = b'\x04'

_byteSuffix = [
    ('B', 'B', 'B'),
    ('KiB', 'K', 'kB'),
    ('MiB', 'M', 'MB'),
    ('GiB', 'G', 'GB'),
    ('TiB', 'T', 'TB'),
    ('PiB', 'P', 'PB'),
    ('EiB', 'E', 'EB'),
]

class ReadableBlockImage(object):
    _io = None
    _filePath = None
    _fileSize = None
    _uuid = None

    _blockSize = None
    _blockCount = None

    _currentBlock = None
    _currentBlockHash = None
    _currentBlockIndex = -1

    _hashAlgorithm = None

    def __init__(self, filePath, blockSize, hashAlgorithm="sha1"):
        self._filePath = filePath
        self._blockSize = blockSize

        self._hashAlgorithm = hashAlgorithm

    @property
    def filePath(self):
        return self._filePath

    @property
    def fileSize(self):
        if self._fileSize is None:
            currentOffset = self._file.tell()

            self._file.seek(0, os.SEEK_END)
            self._fileSize = self._file.tell()
            self._file.seek(currentOffset, os.SEEK_SET)

        return self._fileSize

    @property
    def uuid(self):
        if self._uuid is None:
            try:
                cmd = [ "/sbin/blkid", "-o", "value", "-s", "UUID", self._filePath ]
                self._uuid = subprocess.check_output(cmd).decode("ascii")
                if self._uuid[-1:] == "\n":
                    self._uuid = self._uuid[:-1]
            except Exception:
                self._uuid = ""

        return self._uuid

    @property
    def blockSize(self):
        return self._blockSize

    @property
    def blockCount(self):
        if self._blockCount is None:
            self._blockCount = -(- self.fileSize // self._blockSize)

        return self._blockCount

    @property
    def hashAlgorithm(self):
        return self._hashAlgorithm

    @property
    def closed(self):
        return self._io is None or self._io.closed

    @property
    def _file(self):
        if not self._io:
            self.open()

        return self._io

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.next()
        except EOFError:
            raise StopIteration

    def open(self):
        self.close()

        try:
            self._io = open(self._filePath, "rb")
        except:
            self.close()
            raise

    def close(self):
        if self._io:
            self._io.close()

            self._io = None
            self._fileSize = None
            self._uuid = None

            self._blockCount = None

            self._currentBlock = None
            self._currentBlockHash = None
            self._currentBlockIndex = -1

    def _sync(self):
        if self._currentBlockIndex >= 0:
            currentOffset = min((self._currentBlockIndex + 1) * self._blockSize, self.fileSize)
            self._file.seek(currentOffset, os.SEEK_SET)
        else:
            self._file.seek(0, os.SEEK_SET)

    def next(self):
        self._sync()

        self._currentBlock = self._file.read(self._blockSize)
        self._currentBlockHash = None
        self._currentBlockIndex += 1

        if len(self._currentBlock) > 0:
            return self.read()
        else:
            raise EOFError

    def read(self):
        if self._currentBlock is None:
            if self._currentBlockIndex == -1:
                return None, None

            self.rewind()
            return self.next()

        if self._currentBlockHash is None and len(self._currentBlock) > 0:
            self._currentBlockHash = _hash(self._hashAlgorithm, self._currentBlock)

        return self._currentBlock, self._currentBlockHash

    def rewind(self, blocks=1):
        if not blocks >= 1:
            raise ValueError("Block count must be >= 1")
        if self._currentBlockIndex == -1:
            raise RuntimeError("You can't rewind at the start of the file")

        self._currentBlock = None
        self._currentBlockHash = None
        self._currentBlockIndex = max(self._currentBlockIndex - blocks, -1)

    def reset(self):
        self._currentBlock = None
        self._currentBlockHash = None
        self._currentBlockIndex = -1

    def skip(self, blocks=1):
        if not blocks >= 1:
            raise ValueError("Block count must be >= 1")
        if self._currentBlockIndex + 1 >= self.blockCount:
            raise EOFError

        self._currentBlock = None
        self._currentBlockHash = None
        self._currentBlockIndex = min(self._currentBlockIndex + blocks, self.blockCount)

    def tell(self):
        return self._currentBlockIndex

class WritableBlockImage(ReadableBlockImage):
    def open(self):
        self.close()

        try:
            self._io = open(self._filePath, "r+b")
        except IOError as error:
            if error.errno == errno.ENOENT:
                self._io = open(self._filePath, "w+b")
            else:
                raise
        except:
            self.close()
            raise

    def close(self):
        if self._io:
            self.flush()

        return super(WritableBlockImage, self).close()

    def write(self, bytes):
        if self._currentBlockIndex == -1:
            raise RuntimeError("No block to overwrite")

        self._currentBlockIndex -= 1
        self._sync()

        bytesLength = len(bytes)
        bytesBlockCount = -(- bytesLength // self._blockSize)

        if bytesLength == 0:
            self.truncate()

            self._currentBlock = None
            self._currentBlockHash = None
            return

        self._file.write(bytes)

        lastBlockSize = bytesLength % self._blockSize
        if lastBlockSize == 0:
            lastBlockSize = self._blockSize

            self._fileSize = max(self.fileSize, (self._currentBlockIndex  + 1) * self._blockSize + bytesLength)
            self._blockCount = max(self.blockCount, self._currentBlockIndex + 1 + bytesBlockCount)

            self._currentBlockIndex += bytesBlockCount
        else:
            self._currentBlockIndex += bytesBlockCount
            self.truncate()

        self._currentBlock = bytes[-lastBlockSize:]
        self._currentBlockHash = None

    def seek(self, blocks=1):
        if not blocks >= 1:
            raise ValueError("Block count must be >= 1")

        if self._currentBlockIndex + 1 >= self.blockCount:
            self._sync()
            if self._file.tell() % self._blockSize > 0:
                raise RuntimeError("You must not seek over an incomplete block")

        self._fileSize = max(self.fileSize, (self._currentBlockIndex + 1 + blocks) * self._blockSize)
        self._blockCount = max(self.blockCount, self._currentBlockIndex + 1 + blocks)

        self._currentBlock = None
        self._currentBlockHash = None
        self._currentBlockIndex += blocks

    def truncate(self):
        self._sync()

        self._file.truncate()

        self._fileSize = self._file.tell()
        self._blockCount = None

    def flush(self):
        self._sync()
        self._file.flush()
        os.fsync(self._file.fileno())

class Remote(object):
    _image = None

    @property
    def image(self):
        if not self._image:
            raise RuntimeException("{!r} wasn't initialized yet".format(self.__class__.__name__))

        return self._image

    @property
    def imageInfo(self):
        return {
            "blockSize": self.image.blockSize,
            "hashAlgorithm": self.image.hashAlgorithm,
            "filePath": self.image.filePath,
            "fileSize": self.image.fileSize,
            "uuid": self.image.uuid
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.close()

    def __iter__(self):
        if not self._image:
            raise RuntimeException("{!r} is not iterable yet".format(self.__class__.__name__))

        return self

    def __next__(self):
        try:
            return self.next()
        except EOFError:
            raise StopIteration

    def open(self):
        self.image.open()

    def close(self):
        if self._image:
            self._image.close()

    def next(self):
        return self.image.next()

    def checkIdentifier(self):
        expectedIdentifier = "iblocksync {}\n".format(__version__)
        identifier = sys.stdin.buffer.readline().decode("utf-8")

        if identifier != expectedIdentifier:
            raise ValueError("Expecting identifier {!r}, got {!r}".format(expectedIdentifier, identifier))

    def sendError(self, message, code=None):
        errorDict = { "error": message }
        if code is not None:
            errorDict["errno"] = code

        sys.stdout.buffer.write((json.dumps(errorDict) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    def sendImageInfo(self):
        sys.stdout.buffer.write((json.dumps(self.imageInfo) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    def readImageInfo(self):
        jsonString = sys.stdin.buffer.readline().decode("utf-8")
        return json.loads(jsonString)

    def _sendHashes(self, blockCount):
        for block, hash in itertools.islice(self, blockCount):
            sys.stdout.buffer.write(hash)

        sys.stdout.buffer.flush()

        self.image.reset()

    def _readBlocksToUpdate(self, blockCount):
        blocksToUpdate = []
        for blockIndex in range(blockCount):
            blocksToUpdate.append(self._readSignal())

        return blocksToUpdate

    def _readSignal(self):
        signal = sys.stdin.buffer.read(1)
        if signal == _SIGNAL_ACK:
            return True
        elif signal == _SIGNAL_NAK:
            return False
        else:
            raise ValueError("Expecting signal {!r} or {!r}, got {!r}".format(_SIGNAL_ACK, _SIGNAL_NAK, signal))

    def _getBlocksToUpdate(self, sourceImageInfo, targetImageInfo):
        sourceBlockCount = -(- sourceImageInfo["fileSize"] // sourceImageInfo["blockSize"])
        targetBlockCount = -(- targetImageInfo["fileSize"] // targetImageInfo["blockSize"])
        commonBlockCount = min(sourceBlockCount, targetBlockCount)

        self._sendHashes(commonBlockCount)

        blocksToUpdate = self._readBlocksToUpdate(commonBlockCount)
        blocksToUpdateFiller = itertools.repeat(True, sourceBlockCount - commonBlockCount)

        return itertools.chain(blocksToUpdate, blocksToUpdateFiller)

    def _sendEndOfTransmission(self):
        sys.stdout.buffer.write(_SIGNAL_EOT)
        sys.stdout.buffer.flush()

    def run(self):
        raise NotImplementedError()

class ServeRemote(Remote):
    def __init__(self, imagePath, blockSize, hashAlgorithm):
        self._image = ReadableBlockImage(imagePath, blockSize, hashAlgorithm)

    def run(self, targetImageInfo):
        blocksToUpdate = self._getBlocksToUpdate(self.imageInfo, targetImageInfo)

        for updateBlock in blocksToUpdate:
            if updateBlock:
                localBlock, localHash = self.image.next()
                sys.stdout.buffer.write(localBlock)
            else:
                self.image.skip()

        sys.stdout.buffer.flush()

        self._sendEndOfTransmission()

class ReceiveRemote(Remote):
    def __init__(self, imagePath, blockSize, hashAlgorithm):
        self._image = WritableBlockImage(imagePath, blockSize, hashAlgorithm)

    def run(self, sourceImageInfo):
        blocksToUpdate = self._getBlocksToUpdate(sourceImageInfo, self.imageInfo)

        for updateBlock in blocksToUpdate:
            self.image.seek()

            if updateBlock:
                remoteBlock = self._readBlock()
                self.image.write(remoteBlock)

        self.image.truncate()
        self.image.flush()

        self._sendEndOfTransmission()

    def _readBlock(self):
        block = sys.stdin.buffer.read(self.image.blockSize)
        blockSize = len(block)
        if blockSize != self.image.blockSize:
            raise ValueError("Receiving invalid block, expecting {:,} bytes, got {:,} bytes".format(
                self.image.blockSize, blockSize
            ))

        return block

class Local(threading.Thread):
    _executable = None

    _cmd = None
    _pipe = None

    _imagePath = None
    _imageInfo = None

    _blockSize = None
    _hashAlgorithm = None
    _hashSize = None

    _hashList = None
    _commonBlockCount = None
    _differentBlockCount = None

    _sync = None
    _syncBarrier = threading.Barrier(2)
    _finished = False

    _logStats = None

    def __init__(self, imagePath, blockSize, hashAlgorithm, logStats=None):
        self._blockSize = blockSize
        self._hashAlgorithm = hashAlgorithm
        self._hashSize = _hashSize(hashAlgorithm)

        self._cmd, self._imagePath = self._parseCommand(imagePath)
        self._pipe = subprocess.Popen(
            self._cmd,
            bufsize=-1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            close_fds=True
        )

        if self._pipe.poll() is not None:
            print("Execution failed with {}".format(self._pipe.returncode))
            sys.exit(1)

        self._hashList = []
        self._logStats = logStats

        super(Local, self).__init__()

    @property
    def exec(self):
        return self._executable

    @property
    def rsh(self):
        return [ "ssh" ]

    @property
    def cmd(self):
        return self._cmd

    @property
    def pipe(self):
        return self._pipe

    @property
    def imagePath(self):
        return self._imagePath

    @property
    def imageInfo(self):
        return self._imageInfo

    @property
    def hashList(self):
        return self._hashList

    @property
    def commonBlockCount(self):
        return self._commonBlockCount

    @property
    def differentBlockCount(self):
        return self._differentBlockCount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._close()

    def _close(self):
        if self._pipe and self._pipe.poll() is None:
            self._pipe.kill()

    def _parseCommand(self, imagePath):
        proto = None
        connection = None
        if imagePath[:7] == "file://":
            proto = "file"
            imagePath = imagePath[7:]
        elif imagePath[:6] == "ssh://":
            match = re.match(r"ssh://((?:[^@:/]+@)?[^:/]+(?::\d+)?)/", imagePath)
            if match is not None:
                proto = "ssh"
                connection = match.group(1)
                pathIndex = 6 + len(connection) + 1
                imagePath = imagePath[pathIndex:]
                imagePath = "/" + imagePath if imagePath[:1] != "." else imagePath
        else:
            match = re.match(r"((?:[^@:/]+@)?[^:/]+):", imagePath)
            if match is not None:
                proto = "scp"
                connection = match.group(1)
                pathIndex = len(connection) + 1
                imagePath = imagePath[pathIndex:]

        cmd = None
        if proto is None or proto == "file":
            cmd = []
        elif proto == "ssh" or proto == "scp":
            cmd = self.rsh + [ connection, "--" ]
        else:
            raise ValueError("Invalid protocol {!r}".format(proto))

        cmd += [ self.exec, "--block", str(self._blockSize), "--algo", self._hashAlgorithm, imagePath ]

        return cmd, imagePath

    def sendIdentifier(self):
        self._pipe.stdin.write("iblocksync {}\n".format(__version__).encode("utf-8"))
        self._pipe.stdin.flush()

    def checkImageInfo(self):
        jsonString = self._pipe.stdout.readline()

        if len(jsonString) == 0:
            raise RemoteProcessError(self._executable, "Initialization failed")

        jsonDict = json.loads(jsonString.decode("utf-8"))

        if 'error' in jsonDict:
            errno = jsonDict["errno"] if "errno" in jsonDict else None
            raise RemoteProcessError(self._executable, jsonDict["error"], errno)

        self._imageInfo = jsonDict
        if self._imageInfo["blockSize"] != self._blockSize:
            raise ValueError("Expecting a block size of {:,} bytes, got {:,} bytes".format(
                self._blockSize, self._imageInfo["blockSize"]
            ))
        elif self._imageInfo["hashAlgorithm"] != self._hashAlgorithm:
            raise ValueError("Expecting hash algorithm {}, got {}".format(
                self._hashAlgorithm, self._imageInfo["hashAlgorithm"]
            ))

    def sendImageInfo(self, imageInfo):
        self._pipe.stdin.write((json.dumps(imageInfo) + "\n").encode("utf-8"))
        self._pipe.stdin.flush()

    def prepare(self, sync, sourceBlockCount, targetBlockCount):
        self._sync = sync
        self._commonBlockCount = min(sourceBlockCount, targetBlockCount)
        self._differentBlockCount = sourceBlockCount - self._commonBlockCount

    def run(self):
        try:
            self._hashList = []
            for blockIndex in range(self._commonBlockCount):
                if self._logStats:
                    self._logStats.hitBlockHash(blockIndex, self._executable)

                hash = self._readHash()
                self._hashList.append(hash)

            self._syncBarrier.wait()

            for blockIndex in range(self._commonBlockCount):
                if self._logStats:
                    self._logStats.hitBlock(blockIndex, self._hashList[blockIndex], self._sync.hashList[blockIndex])

                updateBlock = self._hashList[blockIndex] != self._sync.hashList[blockIndex]
                self._pipe.stdin.write(_SIGNAL_ACK if updateBlock else _SIGNAL_NAK)

                if updateBlock:
                    self._differentBlockCount += 1

            self._pipe.stdin.flush()
            self._finished = True
        except Exception:
            self._syncBarrier.abort()
            raise

    def finished(self):
        return self._finished

    def readEndOfTransmission(self):
        signal = self._pipe.stdout.read(1)
        if signal != _SIGNAL_EOT:
            raise ValueError("Expecting EOT signal ({!r}), got {!r}".format(_SIGNAL_EOT, signal))

    def _readHash(self):
        hash = self._pipe.stdout.read(self._hashSize)
        hashSize = len(hash)
        if hashSize != self._hashSize:
            raise ValueError("Receiving invalid hash, expecting {:,} bytes, got {:,} bytes".format(
                self._hashSize, hashSize
            ))

        return hash

    def wait(self):
        self._pipe.wait()

        if self._pipe.returncode > 0:
            raise RemoteProcessError(
                self._executable,
                "Proccess suddenly died with status {}".format(self._pipe.returncode)
            )

class ServeLocal(Local):
    _executable = "iblocksync-serve"

    @property
    def exec(self):
        return os.getenv("IBLOCKSYNC_SOURCE_EXEC", super(ServeLocal, self).exec)

    @property
    def rsh(self):
        if "IBLOCKSYNC_SOURCE_RSH" in os.environ:
            return shlex.split(os.environ["IBLOCKSYNC_SOURCE_RSH"])

        return super(ServeLocal, self).rsh

    def readBlock(self):
        block = self._pipe.stdout.read(self._blockSize)
        blockSize = len(block)
        if blockSize != self._blockSize:
            raise ValueError("Receiving invalid block, expecting {:,} bytes, got {:,} bytes".format(
                self._blockSize, blockSize
            ))

        return block

class ReceiveLocal(Local):
    _executable = "iblocksync-receive"

    @property
    def exec(self):
        return os.getenv("IBLOCKSYNC_TARGET_EXEC", super(ReceiveLocal, self).exec)

    @property
    def rsh(self):
        if "IBLOCKSYNC_TARGET_RSH" in os.environ:
            return shlex.split(os.environ["IBLOCKSYNC_TARGET_RSH"])

        return super(ReceiveLocal, self).rsh

    def sendBlock(self, block):
        self._pipe.stdin.write(block)
        self._pipe.stdin.flush()

class LogStats(threading.Thread):
    _STATE_IDLE = 0
    _STATE_SYNCING = 1
    _STATE_UPDATING = 2
    _STATE_EXCEPTION = 3
    _STATE_STOP = 4

    _logger = None
    _progress = None
    _progressLength = 0

    _server = None
    _receiver = None

    _blocks = None
    _updatedBlocks = None

    _avgSyncTime = None
    _avgUpdateTime = None

    _threadState = _STATE_IDLE
    _threadEvent = threading.Event()

    def __init__(self, verbosity, progress, loggerName="iblocksync"):
        logHandler = logging.StreamHandler(stream=sys.stderr)
        logHandler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))

        self._logger = logging.getLogger(loggerName)
        self._logger.addHandler(logHandler)
        self._logger.setLevel(max(logging.DEBUG, logging.WARNING - verbosity * 10))

        self._progress = progress if progress != "none" else None

        self._blocks = { "source": [], "target": [] }
        self._updatedBlocks = collections.OrderedDict()

        super(LogStats, self).__init__()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.stop()

    @property
    def logger(self):
        return self._logger

    def run(self):
        try:
            while True:
                self._threadEvent.wait()

                if self._threadState == LogStats._STATE_STOP:
                    break
                elif self._threadState == LogStats._STATE_IDLE:
                    self._threadEvent.clear()
                elif self._threadState == LogStats._STATE_SYNCING:
                    self._progressSync()
                    time.sleep(0.5)
                elif self._threadState == LogStats._STATE_UPDATING:
                    self._progressUpdate()
                    time.sleep(0.5)
                else:
                    raise ValueError("Invalid state {!r}".format(self._threadState))
        except:
            self._threadState = LogStats._STATE_EXCEPTION
            raise

    def _set(self, state):
        self._threadState = state
        self._threadEvent.set()

    def stop(self):
        self._set(LogStats._STATE_STOP)

    def init(self, source, target, blockSize, hashAlgorithm):
        self._logger.info("Syncing from {!r} to {!r}".format(source, target))
        self._logger.info("Using a block size of {:,} bytes ({}) and {} hashes".format(
            blockSize,
            _formatBytes(blockSize),
            hashAlgorithm
        ))

    def hitServer(self, server):
        self._server = server

        self._logger.info("Source: {!r}, {}{:,} bytes".format(
            self._server.imageInfo["filePath"],
            "UUID {!r}, ".format(self._server.imageInfo["uuid"]) if self._server.imageInfo["uuid"] != "" else "",
            self._server.imageInfo["fileSize"]
        ))

    def hitReceiver(self, receiver):
        self._receiver = receiver

        self._logger.info("Target: {!r}, {}{:,} bytes".format(
            self._receiver.imageInfo["filePath"],
            "UUID {!r}, ".format(self._receiver.imageInfo["uuid"]) if self._receiver.imageInfo["uuid"] != "" else "",
            self._receiver.imageInfo["fileSize"]
        ))

        if self._server.imageInfo["fileSize"] > 0 and self._receiver.imageInfo["fileSize"] > 0:
            if self._server.imageInfo["fileSize"] != self._receiver.imageInfo["fileSize"]:
                self._logger.warning("Source and target image file sizes don't match")

        if self._server.imageInfo["uuid"] and self._receiver.imageInfo["uuid"]:
            if self._server.imageInfo["uuid"] != self._receiver.imageInfo["uuid"]:
                self._logger.warning("Source and target UUID don't match")

    def startSync(self):
        message = "No blocks to sync"
        if self._server.commonBlockCount > 0:
            self._set(LogStats._STATE_SYNCING)

            message = "Start syncing {:,} of {:,} blocks...".format(
                self._server.commonBlockCount,
                -(- self._server.imageInfo["fileSize"] // self._server.imageInfo["blockSize"]),
            )

        self._logger.info(message)
        if self._progress is not None and self._logger.getEffectiveLevel() > logging.INFO:
            print(message)

    def hitBlockHash(self, blockIndex, remote):
        assert remote in ( "iblocksync-serve", "iblocksync-receive" )
        remote = "source" if remote == "iblocksync-serve" else "target"

        assert blockIndex == len(self._blocks[remote])
        self._blocks[remote].append(time.time())

        if self._progress == "dot" and self._logger.getEffectiveLevel() > logging.DEBUG:
            sys.stdout.write(".")
            sys.stdout.flush()

        self._logger.debug("Receiving hash of {} block #{}...".format(remote, blockIndex))

    def hitBlock(self, blockIndex, sourceHash, targetHash):
        self._logger.debug("Comparing hash of block #{} ({!r} vs. {!r})...".format(blockIndex, sourceHash, targetHash))

    def finishSync(self):
        if self._server.commonBlockCount > 0:
            self._progressSync()
            self._finishProgress()

            serverAverage, serverStartTime, serverFinishTime = self._avgBlocks(self._blocks["source"], None, None)
            serverMessage = "Processed {:,} source blocks in {:,} seconds ({:.2f} blocks/s)".format(
                len(self._blocks["source"]),
                round(serverFinishTime - serverStartTime),
                serverAverage
            )

            receiverAverage, receiverStartTime, receiverFinishTime = self._avgBlocks(self._blocks["target"], None, None)
            receiverMessage = "Processed {:,} target blocks in {:,} seconds ({:.2f} blocks/s)".format(
                len(self._blocks["target"]),
                round(receiverFinishTime - receiverStartTime),
                receiverAverage
            )

            self._logger.info(serverMessage)
            self._logger.info(receiverMessage)
            if self._progress is not None and self._logger.getEffectiveLevel() > logging.INFO:
                print(serverMessage)
                print(receiverMessage)

            self._logger.info("Finished syncing blocks")

    def startUpdate(self):
        message = "No blocks to transmit"
        if self._server.differentBlockCount > 0:
            self._set(LogStats._STATE_UPDATING)

            message = "Start transmitting {:,} of {:,} blocks ({})...".format(
                self._server.differentBlockCount,
                -(- self._server.imageInfo["fileSize"] // self._server.imageInfo["blockSize"]),
                _formatBytes(self._server.differentBlockCount * self._server.imageInfo["blockSize"])
            )

        self._logger.info(message)
        if self._progress is not None and self._logger.getEffectiveLevel() > logging.INFO:
            print(message)

    def startBlockUpdate(self):
        blockIndex = len(self._updatedBlocks)
        self._updatedBlocks[blockIndex] = [ time.time(), None ]

        self._logger.debug("Sending block #{}...".format(blockIndex))

        return blockIndex

    def endBlockUpdate(self, blockIndex):
        self._updatedBlocks[blockIndex][1] = time.time()

        if self._progress == "dot" and self._logger.getEffectiveLevel() > logging.DEBUG:
            sys.stdout.write(".")
            sys.stdout.flush()

        self._logger.debug("Block #{} sent".format(blockIndex))

    def finishUpdate(self):
        if self._server.differentBlockCount > 0:
            self._progressUpdate()
            self._finishProgress()

            average, startTime, finishTime = self._avgUpdates(self._server.imageInfo["blockSize"], None, None)
            message = "Transmitted {:,} blocks ({}) in {:,} seconds ({}/s)".format(
                len(self._updatedBlocks),
                _formatBytes(len(self._updatedBlocks) * self._server.imageInfo["blockSize"]),
                round(finishTime - startTime),
                _formatBytes(average)
            )

            self._logger.info(message)
            if self._progress is not None and self._logger.getEffectiveLevel() > logging.INFO:
                print(message)

            self._logger.info("Finished transmitting blocks")

    def _printProgress(self, progressLine):
        self._progressLength = max(self._progressLength, len(progressLine))
        sys.stdout.write(progressLine.ljust(self._progressLength) + "\r")
        sys.stdout.flush()

    def _finishProgress(self):
        self._set(LogStats._STATE_IDLE)

        self._progressLength = 0
        if self._progress is not None and self._logger.getEffectiveLevel() > logging.DEBUG:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _progressSync(self):
        if self._progress == "full" and self._logger.getEffectiveLevel() > logging.DEBUG:
            self._printProgress("{:,} / {:,} of {:,} blocks processed; {:.2f} blocks/s".format(
                len(self._blocks["source"]),
                len(self._blocks["target"]),
                self._server.commonBlockCount,
                self._avgBlocks(itertools.chain(self._blocks["source"], self._blocks["target"]))[0]
            ))

    def _progressUpdate(self):
        if self._progress == "full" and self._logger.getEffectiveLevel() > logging.DEBUG:
            self._printProgress("{:,} of {:,} blocks transmitted; {}/s".format(
                len(self._updatedBlocks),
                self._server.differentBlockCount,
                _formatBytes(self._avgUpdates(self._server.imageInfo["blockSize"])[0])
            ))

    def _avgBlocks(self, blocks, sampleTime=10, updateEvery=1):
        now = time.time()
        startTime = now
        finishTime = 0

        if updateEvery is not None and self._avgSyncTime is not None:
            if now < self._avgSyncTime[-1] + updateEvery:
                return self._avgSyncTime[:3]

        processedBlocks = 0
        for blockTime in blocks:
            if sampleTime is not None and blockTime < now - sampleTime:
                continue

            processedBlocks += 1
            startTime = min(startTime, blockTime)
            finishTime = max(finishTime, blockTime)

        if processedBlocks > 1:
            avgBlocks = processedBlocks / (finishTime - startTime)
            if updateEvery is not None:
                self._avgSyncTime = ( avgBlocks, startTime, finishTime, now )

            return avgBlocks, startTime, finishTime
        else:
            return 0.0, 0, 0

    def _avgUpdates(self, blockSize, sampleTime=10, updateEvery=1):
        now = time.time()
        startTime = now
        finishTime = 0

        if updateEvery is not None and self._avgUpdateTime is not None:
            if now < self._avgUpdateTime[-1] + updateEvery:
                return self._avgUpdateTime[:3]

        blocks = 0
        duration = 0.0
        for blockIndex, ( blockStartTime, blockEndTime ) in reversed(list(self._updatedBlocks.items())):
            if sampleTime is not None and blockStartTime < now - sampleTime or blockEndTime is None:
                continue

            blocks += 1
            duration += blockEndTime - blockStartTime

            startTime = min(startTime, blockStartTime)
            finishTime = max(finishTime, blockEndTime)

        if blocks > 0:
            avgData = (blocks * blockSize) / duration
            if updateEvery is not None:
                self._avgUpdateTime = ( avgData, startTime, finishTime, now )

            return avgData, startTime, finishTime
        else:
            return 0.0, 0, 0

class RemoteProcessError(Exception):
    """Raised when a remote process encountered a problem."""

    def __init__(self, executable, strerror, errno=None):
        self.executable = executable
        self.strerror = strerror
        self.errno = errno

    def __str__(self):
        message = "[Errno {}] {}".format(self.errno, self.strerror) if self.errno else self.strerror
        return "{}: error: {}\n".format(self.executable, message)

def _formatBytes(bytes):
    for i in reversed(range(len(_byteSuffix))):
        if bytes >= 1024 ** i:
            break

    return "{:,.2f} {}".format(bytes / (1024 ** i), _byteSuffix[i][0])

def _hash(algorithm, data):
    if algorithm == "adler32":
        return struct.pack(">I", zlib.adler32(data) & 0xffffffff)
    elif algorithm == "crc32":
        return struct.pack(">I", zlib.crc32(data) & 0xffffffff)
    else:
        return hashlib.new(algorithm, data).digest()

def _hashSize(algorithm):
    if algorithm == "adler32" or algorithm == "crc32":
        return 4
    else:
        return hashlib.new(algorithm).digest_size

def _getArgumentParser():
    class BytesAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help="", metavar=None):
            try:
                defaultValue = self._parseValue(default)
            except (TypeError, ValueError):
                raise ValueError("default value {!r} is no valid byte string".format(default))

            help = ((help + ".  " if help else "") + "A suffix of B for bytes, K for kilobytes (1,024 bytes), "
                + "M for megabytes (1,024 * 1,024 bytes), G for gigabytes, T for terabytes, P for petabytes or "
                + "E for exabytes is optional.  Use kB/MB/GB/TB/PB/EB for base-10 units (1 kB = 1,000 bytes, "
                + "1 MB = 1,000 * 1,000 bytes, ...).  " + ("Defaults to {} ({:,} bytes)".format(default, defaultValue)))

            super(BytesAction, self).__init__(
                option_strings,
                dest,
                nargs=None,
                default=defaultValue,
                required=required,
                help=help,
                metavar=metavar
            )

        def __call__(self, parser, namespace, values, option_string=None):
            try:
                setattr(namespace, self.dest, self._parseValue(values))
            except (TypeError, ValueError):
                raise argparse.ArgumentError(self, "invalid byte string: {!r}".format(values))

        def _parseValue(self, value):
            value = str(value)
            length = len(value)

            if length > 1:
                for i in reversed(range(len(_byteSuffix))):
                    if length > 2 and value[-2:].lower() == _byteSuffix[i][2].lower():
                        return int(round(float(value[:-2]) * (1000 ** i)))
                    elif value[-1].lower() == _byteSuffix[i][1].lower():
                        return int(round(float(value[:-1]) * (1024 ** i)))

            return int(round(float(value)))

    class CopyrightAction(argparse.Action):
        def __init__(self, option_strings, version=None, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
            if help is None:
                help = "show program's version number and exit"
            super(CopyrightAction, self).__init__(option_strings, dest, default=default, nargs=0, help=help)
            self.version = version

        def __call__(self, parser, namespace, values, option_string=None):
            print("iblocksync {}".format(self.version if self.version is not None else parser.version))
            print("Copyright (C) 2017 Daniel Rudolf")
            print("")
            print("License GPLv3: GNU GPL version 3 only <http://gnu.org/licenses/gpl.html>.")
            print("This is free software: you are free to change and redistribute it.")
            print("There is NO WARRANTY, to the extent permitted by law.")
            print("")
            print("Written by Daniel Rudolf <http://www.daniel-rudolf.de/>")
            sys.exit(0)

    hashAlgorithms = { "adler32", "crc32" } | hashlib.algorithms_guaranteed

    argumentParser = argparse.ArgumentParser(add_help=False)
    argumentGroups = {}

    argumentGroups["args"] = argumentParser.add_argument_group("Arguments")

    argumentGroups["app"] = argumentParser.add_argument_group("Application options")
    argumentGroups["app"].add_argument("-a", "--algo", dest="hashAlgorithm",
        choices=hashAlgorithms, default="sha1", metavar="HASH_ALGO",
        help="Gives the hash algorithm to work with.  The following hash algorithms are supported: "
            + (", ".join(repr(hashAlgorithm) for hashAlgorithm in hashAlgorithms)) + ".  "
            + "Defaults to 'sha1'")
    argumentGroups["app"].add_argument("-b", "--block", dest="blockSize", default="1M", action=BytesAction,
        metavar="BLOCK_SIZE", help="Gives the block size to work with")

    argumentGroups["help"] = argumentParser.add_argument_group("Help options")
    argumentGroups["help"].add_argument("--help", dest="help", action="help",
        help="Display this help and exit")
    argumentGroups["help"].add_argument("--version", version=__version__, action=CopyrightAction,
        help="Output version information and exit")

    return argumentParser, argumentGroups
