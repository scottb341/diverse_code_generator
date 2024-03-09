# !/bin/sh


# zip tokenize the sample

zip -r code -@ < proj_list.txt

cp  code.zip tokenizers/file-level/

cd tokenizers/file-level

python tokenizer.py 

cd .. 
cd ..

# transfer token information into clone detector
cat tokenizers/file-level/files_tokens/* > blocks.file 
cp blocks.file clone-detector/input/dataset

echo "tokenization and clone detection preperation done!"

# set up clone detector 

cat files_tokens/* > blocks.file 
cp blocks.file SourcererCC/clone-detector/input/dataset/

cd clone-detector

python controller.py 

cd ..

mkdir "clone_results"

cat clone-detector/NODE_*/output8.0/query_* > clone_results/results.pairs

echo "clone detection done"

cp -r tokenizers/file-level/files_stats clone_results

echo "ALL DONE!"


