from distutils.core import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="iblocksync",
    version="1.0.1",
    description="Incrementally sync raw disk images.",
    long_description=readme,
    author="Daniel Rudolf",
    url="https://github.com/PhrozenByte/iblocksync",
    license=license,
    py_modules=[ "iblocksync" ],
    scripts=[ "iblocksync", "iblocksync-serve", "iblocksync-receive" ]
)
