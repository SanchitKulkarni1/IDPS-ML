import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Radio,
  BarChart3,
  Lightbulb,
  FileText,
  Users,
  Activity,
  ShieldBan,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Live Detection", href: "/live-detection", icon: Radio },
  { name: "Live Analysis", href: "/live-analysis", icon: Activity },
  { name: "Traffic Analytics", href: "/traffic-analytics", icon: BarChart3 },
  // { name: "Explainability", href: "/explainability", icon: Lightbulb },
  { name: "Prevention", href: "/prevention", icon: ShieldBan },
  { name: "Logs & Reports", href: "/logs", icon: FileText },
  { name: "About Project", href: "/about", icon: Users },
];

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="flex min-h-screen bg-secondary/30">
      {/* Sidebar */}
      <aside className="w-64 bg-card border-r border-border">
        <div className="p-6 border-b border-border">
          <h1 className="text-2xl font-bold text-primary">NIDS/IPS Monitor</h1>
          <p className="text-xs text-muted-foreground mt-1">Intrusion Detection & Prevention</p>
        </div>

        <nav className="p-4 space-y-2">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              end={item.href === "/"}
              className={({ isActive }) =>
                isActive ? "nav-link nav-link-active" : "nav-link nav-link-inactive"
              }
            >
              <item.icon className="h-5 w-5" />
              <span>{item.name}</span>
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
