import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trace-to-Specialist Results",
  description: "A miniature implementation of the enterprise intelligence loop.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
