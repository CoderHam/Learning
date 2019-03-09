// g++ -std=c++14 -g main1.cpp && ./main1 <int choice>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm> // for sort()
#include <set>
using namespace std;

struct node{
  int data=0;
  node *next=nullptr;
  node( int d ) : data{ d }, next{ nullptr } { };
  node() : data{0}, next{nullptr} {};
};

void insert(node *&head, int data){
	node * newNode = new node;
	newNode->data = data;
	newNode->next = head;
	head = newNode;
}

void printList(node *head) {
	while (head){
		cout << head->data << "->";
		head = head->next;
	}
	std::cout << "nullptr" << std::endl;
}

void insertOrdered(node *&head, int data){
	node * newNode = new node, *temp = head;
  newNode->data = data;
  if (head==nullptr){
    head = newNode;
    return;
  }
	while(temp->next)
    temp = temp->next;
  temp->next = newNode;
}

void removeDuplicates(node * head) {
	if (head == nullptr || (head && (head->next == nullptr)))
		return;
	set <int> values;
	node * prev = head;
	node * curr = head->next;
	values.insert(head->data);
	while (curr!=nullptr){
		while (curr && values.find(curr->data) != values.end())
			curr = curr->next;
		prev->next = curr;
		prev = curr;
		if (curr){
			values.insert(curr->data);
			curr = curr->next;
		}
	}
}

void deleleLast(node *head){
  if(head == nullptr)
      return;
  node* temp=head;
  if(head->next == nullptr){
    delete head;
      head = nullptr;
      return;
  }
  while(temp->next && temp->next->next != nullptr)
      temp = temp->next;
  delete temp->next;
  temp->next=nullptr;
}

node* kToLast(node *head, int &k){
  for(int i=0;i<k-1;i++){
    deleleLast(head);
  }
  node* nextNode;
  while(head->next){
    nextNode = head->next;
    head = nextNode;
  }
  return head;
}

void deleteMiddle(node* head){
  if (head == nullptr)
    return;
  if (head->next == nullptr){
    delete head;
    return;
  }
  node *slow_ptr = head, *fast_ptr = head, *prev;
  while (fast_ptr != nullptr && fast_ptr->next != nullptr){
    prev = slow_ptr;
    fast_ptr = fast_ptr->next->next;
    slow_ptr = slow_ptr->next;
  }
  // Since fast moves twice as fast as slow, when fast reaches end, slow is in just after middle
  prev->next=prev->next->next;
  delete slow_ptr;
}

node* partitionList(node *head, int pVal){
  node *smallHead = nullptr, *bigHead = nullptr;
  node *small = nullptr, *big = nullptr;
  if (head==nullptr)
    return nullptr;
  node* curr = head, temp;
  while(curr){
    node *nextNode = curr->next;
    if (curr->data < pVal){
      if (small == nullptr){
          small = curr;
          smallHead = small;
      }
      small->next = curr;
      small = curr;
    }
    else{
      if (big == nullptr){
          big = curr;
          bigHead = big;
      }
      big->next = curr;
      big = curr;
    }
    curr = nextNode;
  }
  // small has all elements < then pVal and big has all elements >=
  small->next = bigHead;
  big->next = nullptr;
  return smallHead;
}

node* sumList(node *h1, node *h2){
  node *s  = nullptr, *head = nullptr;
  int carry = 0, sum=0;
  while(h1){
    node *nextNode = nullptr;
    sum = (h1->data+h2->data+carry)%10;
    carry = (h1->data+h2->data)/10;
    insertOrdered(head,sum);
    h1=h1->next;
    h2=h2->next;
  }
  if (carry==1){
    insertOrdered(head,1);
  }
  return head;
}

bool palindromeList(node *head){
  if (head == nullptr  || (head && (head->next == nullptr))){
    return false;
  }
  // reverse linked list
  node *revHead = nullptr, *nextNode = nullptr;
  node *temp = head;
  while (temp){
    nextNode = temp->next;
    temp->next = revHead;
    revHead = temp;
    temp = nextNode;
  }
  while(head && revHead){
    if (head->data != revHead->data)
      return false;
    head=head->next; revHead=revHead->next;
  }
  return true;
}

node* intersection(node *head1, node*head2){
  node *temp = head2;
  while(head1){
    head2 = temp;
    while(head2){
      if (head1 == head2)
        return head1;
      head2=head2->next;
    }
    head1=head1->next;
  }
  return nullptr;
}

node* loopDetection(node *head){
  set<node*> points;
  while(head){
    if (points.find(head) != points.end())
      points.insert(head);
    else
      return head;
    head=head->next;
  }
  return nullptr;
}

int main(int argc, char* argv[]){
    int progNo = atoi(argv[1]);
		switch (progNo){
      case 1: {
        int arr[] = {1,2,3,5,2,1};
        int len = 6;
        node * head = nullptr;
      	for ( int i = 0; i < len; ++i ) {
      		insertOrdered(head, arr[i]);
      	}
        printList(head);
        removeDuplicates(head);
        printList(head);
        break;
      }
      case 2:{
        int arr[] = {1,2,3,4,5,6};
        int len = 6, k=4;
        node * head = nullptr;
      	for ( int i = 0; i < len; ++i ) {
      		insert(head, arr[i]);
      	}
        cout<<"Original list: ";
        printList(head);
        node* k2l = kToLast(head,k);
        cout<<"4th to last is: "<<k2l->data<<endl;
        break;
      }
      case 3:{
        int arr[] = {1,2,3,5,1};
        int len = 5;
        node * head = nullptr;
      	for ( int i = 0; i < len; ++i ) {
      		insert(head, arr[i]);
      	}
        printList(head);
        deleteMiddle(head);
        printList(head);
        break;
      }
      case 4:{
        int arr[] = {1,2,3,5,3,1};
        int len = 6;
        node * head = nullptr;
      	for ( int i = 0; i < len; ++i ) {
      		insert(head, arr[i]);
      	}
        printList(head);
        node *pHead = partitionList(head,3);
        printList(pHead);
        break;
      }
      case 5:{
        int arr1[] = {1,2,3,4}, arr2[] = {3,4,5,6};
        int len = 4;
        node *head1 = nullptr, *head2 = nullptr;
      	for (int i=0; i<len; i++) {
          insertOrdered(head1, arr1[i]);
      		insertOrdered(head2, arr2[i]);
      	}
        cout<<"List 1: (units is start)"<<endl;
        printList(head1);
        cout<<"List 2:"<<endl;
        printList(head2);
        node *pHead = sumList(head1,head2);
        cout<<"Sum List:"<<endl;
        printList(pHead);
        break;
      }
      case 6:{
        int arr1[] = {1,2,3,4,3,2,1};
        int len = 7;
        node *head = nullptr;
      	for (int i=0; i<len; i++) {
          insertOrdered(head, arr1[i]);
      	}
        printList(head);
        if (palindromeList(head))
          cout<<"Yes, Palindrome."<<endl;
        else
          cout<<"No, not Palindrome."<<endl;
        break;
      }
      case 7:{
        node * list1 = new node(3);
        list1->next = new node(6);
        list1->next->next = new node(9);
        list1->next->next->next = new node(12);
        list1->next->next->next->next = new node(15);
        list1->next->next->next->next->next = new node(18);
        node * list2 = new node(7);
        list2->next = new node(10);
        list2->next->next = list1->next->next->next;
        printList(list1);
        printList(list2);
        node * intersectingNode = intersection(list1, list2);
        if (intersectingNode)
          cout << "Intersecting Node is: " << intersectingNode->data << endl;
        else
          cout << "Lists do not interset." << endl;
        break;
      }
      case 8:{
        node * head = nullptr;
        insert(head ,1);
        insert(head ,2);
        insert(head ,3);
        insert(head ,4);
        insert(head ,5);
        printList(head);
        std::cout << "Inserting loop to connect 5 to 2.\n";
        head->next->next->next->next->next = head->next;
        node *looper = loopDetection(head);
        cout<<"Detected loop at: "<<looper<<" with value: "<<looper->data<<endl;
        break;
      }
      default:{
        cout<<"That program isn't finished yet"<<endl;
        break;
      }
    }
    return 0;
}
