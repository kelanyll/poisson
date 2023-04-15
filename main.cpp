#include <iostream>
#include "Date.hpp"

using namespace std;
using namespace BOOM;

int main() {
    Date date = Date();
    cout << "Hello world it's " << date.str() << "!\n";

    return 0;
}