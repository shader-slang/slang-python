mkdir -p ./tmp
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp

mkdir -p ./bin/
cp ./tmp/bin/windows-x64/release/slang.dll ./bin/slang.dll
cp ./tmp/bin/windows-x64/release/slangc.exe ./bin/slangc.exe
cp ./tmp/bin/linux-x64/release/libslang.so ./bin/libslang.so
cp ./tmp/bin/linux-x64/release/slangc ./bin/slangc
chmod +x ./bin/slangc

echo "content of bin/:"
ls ./bin/

rm $WIN64ZIP
rm $LINUX64ZIP
rm -rf ./tmp/

python3 --version

python3 -m pip install --upgrade pip setuptools wheel build
pip install twine
python3 -m build
