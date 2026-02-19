import { Github, Linkedin, Mail, Database, Shield, Brain, LineChart, Workflow, Bug, Network } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// Commented out for now
/*const teamMembers = [
  {
    name: "Dr. Sarah Chen",
    role: "Project Lead & ML Engineer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah",
    email: "sarah.chen@university.edu",
    linkedin: "#",
    github: "#",
  },
  {
    name: "Michael Rodriguez",
    role: "Backend Developer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Michael",
    email: "m.rodriguez@university.edu",
    linkedin: "#",
    github: "#",
  },
  {
    name: "Priya Sharma",
    role: "Frontend Developer",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Priya",
    email: "priya.sharma@university.edu",
    linkedin: "#",
    github: "#",
  },
  {
    name: "James Wilson",
    role: "Data Scientist",
    image: "https://api.dicebear.com/7.x/avataaars/svg?seed=James",
    email: "j.wilson@university.edu",
    linkedin: "#",
    github: "#",
  },
];*/

const attackTypes = [
  {
    name: "Denial of Service (DoS)",
    description: "Attempts to make network resources unavailable to intended users by overwhelming the target with traffic or requests.",
    icon: Shield,
  },
  {
    name: "Probe Scanning",
    description: "Surveillance and scanning activities to gather information about the network for potential vulnerabilities.",
    icon: Network,
  },
  {
    name: "User to Root (U2R)",
    description: "Unauthorized attempts to escalate privileges from normal user to super-user or admin.",
    icon: Bug,
  },
  {
    name: "Remote to Local (R2L)",
    description: "Unauthorized attempts to gain local access from a remote machine.",
    icon: Workflow,
  },
];

const techStack = [
  {
    title: "Frontend",
    items: ["React", "TypeScript", "Tailwind CSS", "Shadcn/ui"],
  },
  {
    title: "Backend",
    items: ["Python", "Flask", "scikit-learn", "pandas"],
  },
  {
    title: "ML Libraries",
    items: ["LightGBM", "XGBoost", "Random Forest", "scikit-learn"],
  },
  {
    title: "Data Processing",
    items: ["NumPy", "Pandas", "Feature Engineering", "Data Preprocessing"],
  },
  {
    title: "Prevention (Phase 2)",
    items: ["SQLite", "iptables", "Rules Engine", "Auto-Response"],
  },
];

const About = () => {
  return (
    <div className="space-y-8 pb-8">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold text-foreground">About This Project</h1>
        <p className="text-xl text-muted-foreground">Network Intrusion Detection System</p>
      </div>

      {/* Project Abstract */}
      <div className="stat-card">
        <h2 className="text-xl font-semibold mb-4">Abstract</h2>
        <div className="prose prose-sm max-w-none text-muted-foreground space-y-4">
          <p>
            This Network Intrusion Detection System (NIDS) implements a comprehensive machine learning
            approach to identify and classify network intrusions. The system can operate in both binary
            (normal vs attack) and multi-class classification modes, capable of detecting four main
            types of attacks: DoS (Denial of Service), Probe Scanning, U2R (User to Root), and
            R2L (Remote to Local).
          </p>
          <p>
            Our approach leverages multiple advanced machine learning algorithms, with Random Forest
            excelling in multi-class detection and LightGBM performing exceptionally in binary
            classification. The system achieves remarkable accuracy while maintaining practical
            deployment capabilities for real-world network security applications.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Information */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="h-6 w-6 text-primary" />
              <CardTitle>Dataset Overview</CardTitle>
            </div>
            <CardDescription>
              NSL-KDD: A refined version of the KDD'99 dataset, specifically designed for Network Intrusion Detection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Dataset Composition */}
              <div className="space-y-3">
                <div className="text-sm font-medium">Dataset Composition</div>
                <div className="grid gap-2">
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-muted-foreground text-sm">Training Set</span>
                    <span className="font-medium text-sm">~125,973 records</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-muted-foreground text-sm">Testing Set</span>
                    <span className="font-medium text-sm">~22,544 records</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-muted-foreground text-sm">Total Features</span>
                    <span className="font-medium text-sm">41 network characteristics</span>
                  </div>
                </div>
              </div>

              {/* Feature Categories */}
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-2">Basic Features</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    These describe the overall properties of the connection, similar to a phone call record (duration, type, status).
                  </p>
                  <div className="grid gap-2">
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <div className="text-sm font-medium mb-1">Protocol Type</div>
                      <p className="text-sm text-muted-foreground">
                        TCP (web, email), UDP (video calls), ICMP (ping)
                      </p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <div className="text-sm font-medium mb-1">Service & Status</div>
                      <p className="text-sm text-muted-foreground">
                        Application services (HTTP, FTP, SMTP) and connection status flags
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium mb-2">Traffic Features</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Analyze patterns over time, like checking connection frequency in time windows
                  </p>
                  <div className="grid gap-2">
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <div className="text-sm font-medium mb-1">Connection Patterns</div>
                      <p className="text-sm text-muted-foreground">
                        Connections within 2-second windows, same service rates, and traffic patterns
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ML Model Information */}
        <div className="stat-card">
          <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
          <div className="space-y-6 text-sm">
            {/* Multi-class Classification */}
            <div>
              <h3 className="text-sm font-medium mb-3">Multi-class Classification (Random Forest)</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Accuracy</span>
                  <span className="font-medium text-success">99.974%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Precision</span>
                  <span className="font-medium">99.974%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Recall</span>
                  <span className="font-medium">99.974%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">F1 Score</span>
                  <span className="font-medium">99.974%</span>
                </div>
              </div>
            </div>

            {/* Binary Classification */}
            <div>
              <h3 className="text-sm font-medium mb-3">Binary Classification (LightGBM)</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Accuracy</span>
                  <span className="font-medium text-success">99.940%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Precision</span>
                  <span className="font-medium">99.950%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">Recall</span>
                  <span className="font-medium">99.930%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                  <span className="text-muted-foreground">F1 Score</span>
                  <span className="font-medium">99.940%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Attack Types */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {attackTypes.map((attack) => (
          <Card key={attack.name}>
            <CardHeader>
              <div className="flex items-center gap-2">
                <attack.icon className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">{attack.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">{attack.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* System Architecture */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            <CardTitle>System Architecture</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-center">
              {[
                "Data Collection",
                "Preprocessing",
                "Feature Extraction",
                "ML Classification",
                "Alert & Visualization",
              ].map((stage, idx) => (
                <div key={stage}>
                  <div className="bg-primary text-primary-foreground rounded-lg p-4 font-medium text-sm">
                    {stage}
                  </div>
                  {idx < 4 && (
                    <div className="hidden md:block text-primary text-2xl mt-2">→</div>
                  )}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <div className="bg-muted/30 rounded-lg p-6">
                <h3 className="text-sm font-medium mb-4">Model Performance Ranking (Multi-class)</h3>
                <div className="space-y-2 text-sm">
                  {[
                    { model: "Random Forest", score: "99.974%" },
                    { model: "LightGBM", score: "99.972%" },
                    { model: "XGBoost", score: "99.966%" },
                    { model: "Decision Tree", score: "99.924%" }
                  ].map((item, idx) => (
                    <div key={item.model} className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <span className="text-muted-foreground">{idx + 1}. {item.model}</span>
                      <span className="font-medium">{item.score}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-muted/30 rounded-lg p-6">
                <h3 className="text-sm font-medium mb-4">Model Performance Ranking (Binary)</h3>
                <div className="space-y-2 text-sm">
                  {[
                    { model: "LightGBM", score: "99.940%" },
                    { model: "XGBoost", score: "99.935%" },
                    { model: "Random Forest", score: "99.915%" },
                    { model: "Gradient Boosting", score: "99.609%" }
                  ].map((item, idx) => (
                    <div key={item.model} className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <span className="text-muted-foreground">{idx + 1}. {item.model}</span>
                      <span className="font-medium">{item.score}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Stack */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <LineChart className="h-6 w-6 text-primary" />
            <CardTitle>Technical Implementation</CardTitle>
          </div>
          <CardDescription>
            A comprehensive stack of modern technologies powering our NIDS
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techStack.map((section) => (
              <div key={section.title} className="space-y-3">
                <h3 className="text-sm font-medium">{section.title}</h3>
                <ul className="space-y-2">
                  {section.items.map((item) => (
                    <li key={item} className="text-sm text-muted-foreground p-2 bg-muted/50 rounded-lg">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Team 
      <div>
        <h2 className="text-2xl font-semibold mb-6">Our Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member) => (
            <div key={member.name} className="stat-card text-center">
              <img
                src={member.image}
                alt={member.name}
                className="w-24 h-24 rounded-full mx-auto mb-4 bg-muted"
              />
              <h3 className="font-semibold text-foreground">{member.name}</h3>
              <p className="text-sm text-muted-foreground mb-4">{member.role}</p>
              <div className="flex items-center justify-center gap-3">
                <a
                  href={member.email}
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Mail className="h-4 w-4" />
                </a>
                <a
                  href={member.linkedin}
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Linkedin className="h-4 w-4" />
                </a>
                <a
                  href={member.github}
                  className="text-muted-foreground hover:text-primary transition-colors"
                >
                  <Github className="h-4 w-4" />
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>*/}
    </div>
  );
};

export default About;