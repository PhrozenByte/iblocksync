#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# create Python source distribution
[ ! -d _build/dist ] || rm -rf _build/dist
python3 setup.py sdist --dist-dir _build/dist

# Debianize source distribution
cd _build/dist
VERSION="$(find . -mindepth 1 -maxdepth 1 -name 'iblocksync-*.tar.gz' \
    | sed -e 's/^\.\/iblocksync-\(.*\)\.tar\.gz$/\1/g')"
ln -s "iblocksync-$VERSION.tar.gz" "iblocksync_$VERSION.orig.tar.gz"
tar xfz "iblocksync-$VERSION.tar.gz"
cp -R ../debian "iblocksync-$VERSION"

# build package
cd "iblocksync-$VERSION"
dpkg-buildpackage -rfakeroot -uc -us

# success
cd ../../..
PACKAGE="$(find _build/dist -mindepth 1 -maxdepth 1 -name 'iblocksync_*_all.deb')"

echo
echo "Success! Run \`dpkg -i \"$PACKAGE\"\` to install the package..."
