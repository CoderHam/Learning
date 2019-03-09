// g++ -std=c++14 -g main2.cpp && ./main2 <int choice>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm> // for sort()
#include <set>
using namespace std;

bool isUniqueChar(const string &str){
  vector<bool> char_set(255);
  for (auto c : str){
    if (char_set[c])
      return false;
    char_set[c]=true;
  }
  return true;
}

bool isPermutation(const string &str1, const string &str2){
  if (str1.length() != str2.length())
    return false;
  int counter[256]={0};
  for (auto c : str1){
    counter[c]++;
  }
  for (auto c : str2){
    counter[c]--;
    if(counter[c]<0)
      return false;
  }
  return true;
}

void URLify(string &str, int len){
  int numSpaces = 0, i=0, j=0;
  for (i=0; i<len; i++){
    if (str[i] ==' ')
      numSpaces++;
  }
  int extendLen = len + (2 * numSpaces) ;
  i = extendLen-1, j=0;
  for (j=len-1; j>=0; --j){
    if (str[j]==' '){
      str[i--] = '0';
      str[i--] = '2';
      str[i--] = '%';
    }
    else
      str[i--] = str[j];
  }
}

int ignoreCase(char c){
  if('a' <= c && c <= 'z')
    return c - 'a';
  else if('A' <= c && c <= 'Z')
      return c - 'A';
  return -1;
}

bool palindromePermutation(string &str){
  int counter[26]={0};
  for (auto c : str){
    if(ignoreCase(c)!=-1)
      counter[ignoreCase(c)]++;
  }
  int oddCount;
  for(int count : counter){
    if (count%2==1)
      oddCount++;
  }
  if (oddCount<=1)
    return true;
  return false;
}

bool oneAway(string &s1, string &s2){
  int i=0;
  bool flag=false;
  if (s1.length()>s2.length())
    s1.swap(s2);
  // pale, bale
  if (s1.length()==s2.length()){
    for(i=0;i<s1.length();i++){
      if (s1[i]==s2[i])
        continue;
      else if (flag)
        return false;
      else
        flag=true;
    }
  }
  // pale, ple OR ple, pale
  else if (s1.length()+1==s2.length()){
    int i=0,j=0;
    while(i<s2.length() && j<s1.length()){
      cout<<s1[i]<<"<->"<<s2[j]<<endl;
      if (s1[i]==s2[j]){
          i++;j++;
      }
      else if (flag){
        return false;
      }
      else{
        cout<<"switch"<<endl;
        flag = true;
        i++;
      }
    }
  }
  return true;
}

string stringCompression(string &str){
  string comp = "";
  int counter[26]={0};
  for (auto c : str){
    if(ignoreCase(c)!=-1)
      counter[ignoreCase(c)]++;
  }
  for(int i=0;i<26;i++){
    if (counter[i]!=0)
      comp = comp + (char)((int)'a'+i) + (char)(48+counter[i]);
  }
  return comp.length()<str.length()?comp:str;
}

void rotateMatrix90(int **mat, int N){
  int temp=0;
  for (int i=0;i<N/2;i++){
    for (int j=0;j<N-i-1;j++){
      temp = mat[i][j];
      // move values from right to top
      mat[i][j] = mat[j][N-1-i];
      // move values from bottom to right
      mat[j][N-1-i] = mat[N-1-i][N-1-j];
      // move values from left to bottom
      mat[N-1-i][N-1-j] = mat[N-1-j][i];
      // assign temp to left
      mat[N-1-j][i] = temp;
    }
  }
}

void displayMatrix(int **mat, int N){
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++)
        cout<<mat[i][j]<<"\t";
      cout<<endl;
    }
}

void zeroMatrix(int **mat, int N){
  set<int> set0I;
  set<int> set0J;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (mat[i][j]==0){
        set0I.insert(i);
        set0J.insert(j);
      }
    }
  }
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (set0I.find(i) != set0I.end() || set0J.find(j) != set0J.end()){
        mat[i][j]=0;
      }
    }
  }
}

bool stringRotation(string &s1, string &s2){
  if (s1.compare(s2))
    return true;
  if (s1.length()!=s2.length())
    return false;
  string twiceS1 = s1+s1;
  return twiceS1.find(s2) != string::npos;
}

int main(int argc, char* argv[]){
    int progNo = atoi(argv[1]);
		switch (progNo){
      case 1: {
        vector<string> words = {"hello", "welcome", "to", "my", "test"};
    		for (auto word : words)
    		{
    			cout << word <<": "<< boolalpha << isUniqueChar(word) <<endl;
    		}
        break;
      }
      case 2:{
        string str1="testest", str2="ttestex";
        if(isPermutation(str1, str2))
          cout << str1 <<" and "<< str2 << " are permutations of each other" << endl;
        else
          cout << str1 <<" and "<< str2 << " are not permutations of each other" << endl;
        break;
      }
      case 3:{
        string str = "Mr John Smith    ";                       //String with extended length ( true length + 2* spaces)
        std::cout << "Actual string  : \"" << str << "\"\n";
        URLify(str,13);                                        //Length of "Mr John Smith" = 13
        std::cout << "URLified string: \"" << str << "\"\n";
        break;
      }
      case 4:{
        string pali = "Rats live on no evil star";
        string isPermutationPalin = palindromePermutation(pali) ? "Yes, it is a Palindrome Permutation" : "No, it is not a Palindrome Permutation";
        cout << isPermutationPalin << endl;
        break;
      }
      case 5:{
        string s1="play",s2="blay";
        if(oneAway(s1,s2))
          cout<<"One edit away."<<endl;
        else
          cout<<"Not one edit away."<<endl;
        break;
      }
      case 6:{
        string str = "aabbbbc";
        cout<<"Compressed String: "<<stringCompression(str)<<endl;
        break;
      }
      case 7:{
        int matrix[3][3] = { {1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9} };
        int *mat[3];
        for (int x = 0; x < 3; x++) mat[x] = matrix[x];
        cout<<"Input Matrix:"<<endl;
        displayMatrix(mat,3);
        rotateMatrix90(mat,3);
        cout<<"Rotated Matrix:"<<endl;
        displayMatrix(mat,3);
        break;
      }
      case 8:{
        int matrix[3][3] = { {1, 0, 3},
                             {4, 5, 0},
                             {7, 8, 9} };
        int *mat[3];
        for (int x = 0; x < 3; x++) mat[x] = matrix[x];
        cout<<"Input Matrix:"<<endl;
        displayMatrix(mat,3);
        zeroMatrix(mat,3);
        cout<<"Zeroed Matrix:"<<endl;
        displayMatrix(mat,3);
        break;
      }
      case 9:{
        string s1="player",s2="erplay";
        if(stringRotation(s1,s2))
          cout<<"Yes, "<<s1<<" and "<<s2<<" are string rotations."<<endl;
        else
          cout<<"No, "<<s1<<" and "<<s2<<" are not string rotations."<<endl;
        break;
      }
      default:{
        cout<<"That program isn't finished yet"<<endl;
        break;
      }
    }
    return 0;
}
