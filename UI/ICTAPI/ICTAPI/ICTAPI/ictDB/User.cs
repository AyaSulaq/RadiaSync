using System;
using System.Collections.Generic;
using ICTAPI.ictDB;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.OpenApi;
using Microsoft.AspNetCore.Http.HttpResults;

namespace ICTAPI.ictDB;

public partial class User
{
    public int Id { get; set; }

    public string FName { get; set; } = null!;

    public string LName { get; set; } = null!;

    public string Email { get; set; } = null!;

    public string Password { get; set; } = null!;

    public string? CreatedAt { get; set; }

    public string? EditedAt { get; set; }

    public string? Token { get; set; }

    public virtual ICollection<WorkContent> WorkContents { get; set; } = new List<WorkContent>();
}

