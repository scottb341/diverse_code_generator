
# !/bin/sh/ 

rm code.zip

rm -r clone_results

cd tokenizers/file-level

sh cleanup.sh

cd ..

cd ..

cd clone-detector

sh cleanup.sh

echo "cleanup complete!"
