#include<cassert>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
using namespace tensorflow;

int checkpoint = 0;

bool check(Status status) {
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return false;
	}
	++checkpoint;
	//std::cout << "Passed checkpoint " << checkpoint << endl;
	return true;
}

int main(int argc, char* argv[]) {

  // Initialize a tensorflow session
  Session* session;
  assert(check(NewSession(SessionOptions(), &session)));

  // Read in the protobuf graph
  GraphDef graph_def;
  assert(check(ReadBinaryProto(Env::Default(), "graph.pb", &graph_def)));

  // Add the graph to the session
  assert(check(session->Create(graph_def)));

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session
  assert(check(session->Run({}, {}, {"init"}, &outputs)));

  for (int i = 0; i < 201; ++i) {
	  assert(check(session->Run({}, {}, {"train"}, &outputs)));
	  if (i % 20 == 0) {
		  assert(check(session->Run({}, {"W"}, {}, &outputs)));
		  auto W = outputs[0].scalar<float>();
		  assert(check(session->Run({}, {"b"}, {}, &outputs)));
		  auto b = outputs[0].scalar<float>();
		  cout << "#step : " << i << ' ' << "W = " << W() << ' ' << "b = " << b() << endl;
	  }
  }

  // Free any resources used by the session
  session->Close();
  return 0;
}

