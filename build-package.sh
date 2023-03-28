python -m pip install --upgrade pip setuptools wheel
pip install twine

ls ./
mkdir -p ./tmp
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp

pip install build

mkdir -p ./bin/
cp ./tmp/bin/windows-x64/release/slang.dll ./bin/slang.dll
cp ./tmp/bin/windows-x64/release/slangc.exe ./bin/slangc.exe
cp ./tmp/bin/linux-x64/release/libslang.so ./bin/libslang.so
cp ./tmp/bin/linux-x64/release/slangc ./bin/slangc
python3 -m build
