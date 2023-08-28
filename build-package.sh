mkdir -p ./tmp
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp

mkdir -p ./slangpy/bin/
cp ./tmp/bin/windows-x64/release/slang.dll ./slangpy/bin/slang.dll
cp ./tmp/bin/windows-x64/release/slangc.exe ./slangpy/bin/slangc.exe
cp ./tmp/bin/linux-x64/release/libslang.so ./slangpy/bin/libslang.so
cp ./tmp/bin/linux-x64/release/slangc ./slangpy/bin/slangc
chmod +x ./slangpy/bin/slangc

echo "content of ./slangpy/bin/:"
ls ./slangpy/bin/

rm $WIN64ZIP
rm $LINUX64ZIP
rm -rf ./tmp/

python3 --version

python -m pip install --upgrade pip
pip install build hatchling

python -m build